#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

int main(int argc, char *argv[]){
    if (argc != 2){
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        exit(1);
    }

    int n= atoi(argv[1]);
    for(int i =0; i<n; i++){
        pid_t pid = fork();
        if (pid < 0){
            perror("fork failed");
            exit(1);
        }
        if (pid == 0){
            printf("Fils #%d,  pid =%d, ppid=%d\n",
                   i+1,getpid(),getppid());
            exit(0);
        }
    }
    return 0;
}
