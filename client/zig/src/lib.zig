const std = @import("std");

pub const DEFAULT_BASE_URL = "http://localhost:8000";

pub const NetworkConfig = struct {
    num_in: u32,
    layer_sizes: []const u32,
    unit_activation: ?[]const u8 = null,
};

pub const Network = struct {
    id: []const u8,
    config: NetworkConfig,
    parameters: ?[]const f64 = null,
};

pub const FitResult = struct {
    epochs: u32,
    losses: []const f64,
};

const PredictResponse = struct {
    outputs: []const []const f64,
};

pub const ClientError = struct {
    kind: Kind,
    message: []const u8,
    status_code: u16 = 0,

    pub const Kind = enum { api, transport, encode, decode };

    pub fn format(
        self: ClientError,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        switch (self.kind) {
            .api => if (self.message.len == 0)
                try writer.print("scalarflow: request failed with status {d}", .{self.status_code})
            else
                try writer.print(
                    "scalarflow: request failed with status {d}: {s}",
                    .{ self.status_code, self.message },
                ),
            .transport => try writer.print("scalarflow: send request: {s}", .{self.message}),
            .encode => try writer.print("scalarflow: encode request: {s}", .{self.message}),
            .decode => try writer.print("scalarflow: decode response: {s}", .{self.message}),
        }
    }
};

pub fn Outcome(comptime T: type) type {
    return union(enum) {
        ok: T,
        err: ClientError,
    };
}

const RequestError = error{ Api, Transport, Encode, Decode, OutOfMemory };

pub const Client = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    http: std.http.Client,
    base_url: []const u8,
    max_response_bytes: usize = 16 * 1024 * 1024,
    last_error: ?ClientError = null,

    pub fn init(allocator: std.mem.Allocator, base_url: []const u8) !Client {
        var arena = std.heap.ArenaAllocator.init(allocator);

        var url = base_url;
        if (std.mem.trim(u8, url, " \t\r\n").len == 0) {
            url = DEFAULT_BASE_URL;
        }
        url = std.mem.trimRight(u8, url, "/");

        const owned_base_url = try arena.allocator().dupe(u8, url);

        return .{
            .allocator = allocator,
            .arena = arena,
            .http = std.http.Client{ .allocator = allocator },
            .base_url = owned_base_url,
        };
    }

    pub fn deinit(self: *Client) void {
        self.http.deinit();
        self.arena.deinit();
    }

    /// Releases all memory returned by previous calls. Any slices handed back by
    /// this client become invalid after this call.
    pub fn reset(self: *Client) void {
        _ = self.arena.reset(.retain_capacity);
    }

    pub fn createNetwork(self: *Client, config: NetworkConfig) Outcome(Network) {
        const body = self.encodeBody(config) catch |err| return .{ .err = self.errFor(err) };
        const data = self.request(.POST, "/networks", body) catch |err| return .{ .err = self.errFor(err) };
        const network = self.decode(Network, data) catch |err| return .{ .err = self.errFor(err) };
        return .{ .ok = network };
    }

    pub fn getNetwork(self: *Client, id: []const u8) Outcome(Network) {
        const path = self.networkPath("/networks/{s}", id) catch |err| return .{ .err = self.errFor(err) };
        const data = self.request(.GET, path, null) catch |err| return .{ .err = self.errFor(err) };
        const network = self.decode(Network, data) catch |err| return .{ .err = self.errFor(err) };
        return .{ .ok = network };
    }

    pub fn deleteNetwork(self: *Client, id: []const u8) Outcome(void) {
        const path = self.networkPath("/networks/{s}", id) catch |err| return .{ .err = self.errFor(err) };
        _ = self.request(.DELETE, path, null) catch |err| return .{ .err = self.errFor(err) };
        return .{ .ok = {} };
    }

    pub fn predict(
        self: *Client,
        id: []const u8,
        inputs: []const []const f64,
    ) Outcome([]const []const f64) {
        const path = self.networkPath("/networks/{s}/predict", id) catch |err| return .{ .err = self.errFor(err) };
        const body = self.encodeBody(.{ .inputs = inputs }) catch |err| return .{ .err = self.errFor(err) };
        const data = self.request(.POST, path, body) catch |err| return .{ .err = self.errFor(err) };
        const response = self.decode(PredictResponse, data) catch |err| return .{ .err = self.errFor(err) };
        return .{ .ok = response.outputs };
    }

    pub fn fit(
        self: *Client,
        id: []const u8,
        inputs: []const []const f64,
        outputs: []const []const f64,
        epochs: u32,
    ) Outcome(FitResult) {
        const path = self.networkPath("/networks/{s}/fit", id) catch |err| return .{ .err = self.errFor(err) };
        const body = self.encodeBody(.{
            .inputs = inputs,
            .outputs = outputs,
            .epochs = epochs,
        }) catch |err| return .{ .err = self.errFor(err) };
        const data = self.request(.POST, path, body) catch |err| return .{ .err = self.errFor(err) };
        const result = self.decode(FitResult, data) catch |err| return .{ .err = self.errFor(err) };
        return .{ .ok = result };
    }

    fn networkPath(self: *Client, comptime suffix: []const u8, id: []const u8) RequestError![]const u8 {
        const a = self.arena.allocator();
        const segment = encodeSegment(a, id) catch {
            self.last_error = .{ .kind = .encode, .message = "out of memory" };
            return error.OutOfMemory;
        };
        return std.fmt.allocPrint(a, suffix, .{segment}) catch {
            self.last_error = .{ .kind = .encode, .message = "out of memory" };
            return error.OutOfMemory;
        };
    }

    fn encodeBody(self: *Client, value: anytype) RequestError![]const u8 {
        return std.json.stringifyAlloc(self.arena.allocator(), value, .{
            .emit_null_optional_fields = false,
        }) catch |err| {
            self.last_error = .{ .kind = .encode, .message = @errorName(err) };
            return error.Encode;
        };
    }

    fn decode(self: *Client, comptime T: type, data: []const u8) RequestError!T {
        return std.json.parseFromSliceLeaky(T, self.arena.allocator(), data, .{
            .ignore_unknown_fields = true,
        }) catch |err| {
            self.last_error = .{ .kind = .decode, .message = @errorName(err) };
            return error.Decode;
        };
    }

    fn request(
        self: *Client,
        method: std.http.Method,
        path: []const u8,
        body: ?[]const u8,
    ) RequestError![]const u8 {
        self.last_error = null;
        const a = self.arena.allocator();

        const url = std.fmt.allocPrint(a, "{s}{s}", .{ self.base_url, path }) catch {
            self.last_error = .{ .kind = .encode, .message = "out of memory" };
            return error.OutOfMemory;
        };
        const uri = std.Uri.parse(url) catch |err| return self.transportError(err);

        var header_buffer: [16 * 1024]u8 = undefined;
        var header_storage: [2]std.http.Header = undefined;
        var header_count: usize = 0;
        header_storage[header_count] = .{ .name = "Accept", .value = "application/json" };
        header_count += 1;
        if (body != null) {
            header_storage[header_count] = .{ .name = "Content-Type", .value = "application/json" };
            header_count += 1;
        }

        var req = self.http.open(method, uri, .{
            .server_header_buffer = &header_buffer,
            .extra_headers = header_storage[0..header_count],
        }) catch |err| return self.transportError(err);
        defer req.deinit();

        if (body) |payload| {
            req.transfer_encoding = .{ .content_length = payload.len };
            req.send() catch |err| return self.transportError(err);
            req.writeAll(payload) catch |err| return self.transportError(err);
            req.finish() catch |err| return self.transportError(err);
        } else {
            req.send() catch |err| return self.transportError(err);
            req.finish() catch |err| return self.transportError(err);
        }
        req.wait() catch |err| return self.transportError(err);

        const data = req.reader().readAllAlloc(a, self.max_response_bytes) catch |err| {
            return self.transportError(err);
        };

        const status_code = @intFromEnum(req.response.status);
        if (status_code < 200 or status_code >= 300) {
            self.last_error = .{
                .kind = .api,
                .status_code = @intCast(status_code),
                .message = errorMessage(a, data),
            };
            return error.Api;
        }

        return data;
    }

    fn transportError(self: *Client, err: anyerror) RequestError {
        self.last_error = .{ .kind = .transport, .message = @errorName(err) };
        return error.Transport;
    }

    fn errFor(self: *Client, err: RequestError) ClientError {
        if (self.last_error) |detail| return detail;
        return switch (err) {
            error.OutOfMemory => .{ .kind = .encode, .message = "out of memory" },
            error.Transport => .{ .kind = .transport, .message = "unknown transport error" },
            error.Encode => .{ .kind = .encode, .message = "unknown encode error" },
            error.Decode => .{ .kind = .decode, .message = "unknown decode error" },
            error.Api => .{ .kind = .api, .message = "" },
        };
    }
};

fn errorMessage(allocator: std.mem.Allocator, text: []const u8) []const u8 {
    if (std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{})) |value| {
        switch (value) {
            .object => |object| {
                if (object.get("error")) |field| {
                    switch (field) {
                        .string => |message| return message,
                        else => {},
                    }
                }
                if (object.get("message")) |field| {
                    switch (field) {
                        .string => |message| return message,
                        else => {},
                    }
                }
            },
            else => {},
        }
    } else |_| {}

    return std.mem.trim(u8, text, " \t\r\n");
}

fn encodeSegment(allocator: std.mem.Allocator, segment: []const u8) ![]const u8 {
    var encoded = std.ArrayList(u8).init(allocator);

    for (segment) |byte| {
        switch (byte) {
            'A'...'Z', 'a'...'z', '0'...'9', '-', '_', '.', '~' => try encoded.append(byte),
            else => try encoded.writer().print("%{X:0>2}", .{byte}),
        }
    }

    return encoded.toOwnedSlice();
}

test "encodeSegment escapes reserved characters" {
    const allocator = std.testing.allocator;
    const encoded = try encodeSegment(allocator, "a/b c");
    defer allocator.free(encoded);
    try std.testing.expectEqualStrings("a%2Fb%20c", encoded);
}

test "encodeSegment keeps unreserved characters" {
    const allocator = std.testing.allocator;
    const encoded = try encodeSegment(allocator, "abc-_.~");
    defer allocator.free(encoded);
    try std.testing.expectEqualStrings("abc-_.~", encoded);
}

test "errorMessage prefers the error field" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const message = errorMessage(arena.allocator(), "{\"error\":\"bad\",\"message\":\"worse\"}");
    try std.testing.expectEqualStrings("bad", message);
}

test "errorMessage falls back to the message field" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const message = errorMessage(arena.allocator(), "{\"message\":\"worse\"}");
    try std.testing.expectEqualStrings("worse", message);
}

test "errorMessage falls back to the raw body" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const message = errorMessage(arena.allocator(), "  plain failure  ");
    try std.testing.expectEqualStrings("plain failure", message);
}

test "client uses the default base url for blank input" {
    var client = try Client.init(std.testing.allocator, "   ");
    defer client.deinit();

    try std.testing.expectEqualStrings(DEFAULT_BASE_URL, client.base_url);
}

test "client trims trailing slashes" {
    var client = try Client.init(std.testing.allocator, "http://example.com///");
    defer client.deinit();

    try std.testing.expectEqualStrings("http://example.com", client.base_url);
}

test "client error display with and without a message" {
    const allocator = std.testing.allocator;

    const with_message = ClientError{
        .kind = .api,
        .status_code = 400,
        .message = "bad request",
    };
    const rendered_with = try std.fmt.allocPrint(allocator, "{}", .{with_message});
    defer allocator.free(rendered_with);
    try std.testing.expectEqualStrings(
        "scalarflow: request failed with status 400: bad request",
        rendered_with,
    );

    const without_message = ClientError{ .kind = .api, .status_code = 404 };
    const rendered_without = try std.fmt.allocPrint(allocator, "{}", .{without_message});
    defer allocator.free(rendered_without);
    try std.testing.expectEqualStrings(
        "scalarflow: request failed with status 404",
        rendered_without,
    );
}

test "client error display for transport, encode and decode" {
    const allocator = std.testing.allocator;

    const cases = [_]struct { value: ClientError, expected: []const u8 }{
        .{
            .value = .{ .kind = .transport, .message = "boom" },
            .expected = "scalarflow: send request: boom",
        },
        .{
            .value = .{ .kind = .encode, .message = "boom" },
            .expected = "scalarflow: encode request: boom",
        },
        .{
            .value = .{ .kind = .decode, .message = "boom" },
            .expected = "scalarflow: decode response: boom",
        },
    };

    for (cases) |case| {
        const rendered = try std.fmt.allocPrint(allocator, "{}", .{case.value});
        defer allocator.free(rendered);
        try std.testing.expectEqualStrings(case.expected, rendered);
    }
}
