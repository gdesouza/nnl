package main

/*
#cgo CFLAGS: -I..
#cgo LDFLAGS: -L.. -lsimple_mlp -lm
#include "simple_mlp.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func main() {
	inputSize := int(C.simple_mlp_input_size())
	outputSize := int(C.simple_mlp_output_size())
	fmt.Printf("Input size:  %d\n", inputSize)
	fmt.Printf("Output size: %d\n", outputSize)

	input := [4]C.float{1.0, 2.0, 3.0, 4.0}
	output := make([]C.float, outputSize)

	rc := C.simple_mlp_infer(unsafe.Pointer(&input[0]), unsafe.Pointer(&output[0]))
	if rc != 0 {
		panic(fmt.Sprintf("inference failed with code %d", rc))
	}

	fmt.Printf("Output: [")
	for i, v := range output {
		if i > 0 {
			fmt.Printf(", ")
		}
		fmt.Printf("%.6f", float64(v))
	}
	fmt.Println("]")

	predicted := 0
	for i := 1; i < len(output); i++ {
		if output[i] > output[predicted] {
			predicted = i
		}
	}
	fmt.Printf("Predicted class: %d\n", predicted)
}
