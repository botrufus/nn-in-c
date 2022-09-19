main:
	gcc *.c -o main -Wall -lm -I .

debug:
	gcc *.c -g -o main -lm

check:
	gcc layer.c network.c main.c -o main -Wall -lm -I . -fsanitize=address -fno-omit-frame-pointer

clean:
	rm main
