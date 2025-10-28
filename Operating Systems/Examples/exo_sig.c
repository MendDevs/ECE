#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

// 1 : Catching ctrl + c (sigint)
void handle_sigint(int sig)
{
    printf("Caught signal %d (SIGINT). Exiting...\n", sig);
    exit(0);
}

int main()
    {
    signal(SIGINT, handle_sigint);
    while (1)
    {
        printf("working...\n");
        sleep(1);
    }
    return 0;
}