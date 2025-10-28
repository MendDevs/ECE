#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <signal.h>
#define N 5

void action(int signum){
    printf("Capture de SIGCONT %d\n", getpid());
}
/*0*/
int main()
{
    pid_t pid[N];
    int i;
    /*1*/
    signal(SIGCONT,action);
    for (i = 0; i < N; i++)
        if ((pid[i] = fork()) == 0)
        {
            pause();
            while(1)
            {printf("ici fils %d \n",i);

        } /*3*/
    /*2*/
    while (1)
        printf("ici fils %d ", i);
    /*3*/
}