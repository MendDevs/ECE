#include<stdio.h>
#include<stdlib.h>
#include<sys/wait.h>
#include<signal.h>
#include<sys/types.h>
#include<unistd.h>

int main(){

    pid_t pid=fork();
    int status;
    if(pid == 0){
        while(1) sleep(1);
        exit(0);
    }
    if (kill(pid,0) == -1){
        printf("fils %d inexistant.\n",pid);
        exit(1);
    }
    else{
        printf("Envoi du signal SIGUSR1 au processus %d \n",pid);
        kill(pid,SIGUSR1);
    }
    pid=waitpid(pid, &status, 0);
    printf("Status du fils %d : %d \n",pid,status);

}