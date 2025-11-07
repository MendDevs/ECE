/*
    A program that takes a file as an argument and displays:
    1. 1. its inode number ;
    2. the number of links it has ;
    3. its owner ;
    4. his group;     
    5. its size; 
    6. its type.
You will use the stat function and then modify the program to use the fstat function.
*/

#include <unistd.h>
#include <stdio.h>

