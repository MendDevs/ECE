#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

int main()
{
    pid_t pid1=fork(), pid2=fork(), pid3=fork();

    if(pid1 == 0){
        printf("Fils1 (pid=%d) : je deviens zombie puis orphelin\n",getpid());
        exit(0); //zombie -> pere ne fait pas de wait
    }

    if(pid2 ==0){
        sleep(5);
        printf("Fils2 (pid=%d) : je deviens orphelin puis zombie\n",getpid());
        exit(0);
    }

    if(pid3==0){
        printf("Fils3 (pid=%d) : je deviens zombie mais pas orphelin\n", getpid());
        exit(0);
    }
    sleep(2);
    printf("Père (pid=%d) : je termine sans wait pour fils1 et fils3.\n", getpid());
    return 0;
}