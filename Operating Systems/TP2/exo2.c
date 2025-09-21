#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(){
    pid_t pid1 = fork();


    if (pid1 == 0){ // fils 1
    printf("Count Process 1\n");
        for(int i = 1; i<=50; i++){
            printf("%d ", i);
        }
        printf("\n\n");
        exit(0);
    }else{

        pid_t pid2 = fork();
        if(pid2 == 0){ // fils2
            printf("Count Process 2\n");
        for(int i = 51; i <= 100; i++){
            printf("%d ", i);
        }
         printf("\n");
         exit(0);
        }
    }

    // version modified:
    /*
        pid_t pid1 = fork();
    if (pid1 == 0) {
        for (int i = 1; i <= 50; i++) printf("%d ", i);
        exit(0);
    } else {
        wait(NULL); // attendre le fils 1
        pid_t pid2 = fork();
        if (pid2 == 0) {
            for (int i = 51; i <= 100; i++) printf("%d ", i);
            exit(0);
        }
        wait(NULL);
    }
    */

    return 0;
}
