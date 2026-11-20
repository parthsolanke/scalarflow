package main

import (
	"log"

	scalarflow "scalarflow"
)

func main() {
	server := scalarflow.NewPongServer(scalarflow.DefaultPongAddr)
	log.Printf("ScalarFlow pong endpoint listening on http://%s/pong", scalarflow.DefaultPongAddr)
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
