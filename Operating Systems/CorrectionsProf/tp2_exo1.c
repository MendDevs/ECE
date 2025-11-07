#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main (int argc, char **argv){
  if(argc != 1){
    fprintf(stderr, "Sytax Error\n Usage : %s",argv[0]);
    exit(1);
  }

  pid_t pid1 = fork();  
  if(pid1 == 0){
    printf("Fils #1: \n");
    for (int i =1; i<=50; i++){ 
      printf("%d ",i);
    }
    exit(0);
   }

    printf("\n\n");
    pid_t pid2 = fork();
    printf("Fils #2: \n");
    if(pid2 == 0){
    for (int j = 51; j <= 100; j++)
    {
      printf("%d ", j);
    }
     exit(0);
    }
  else {exit(1);}
  return 0;
}
