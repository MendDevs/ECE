#include<stdio.h>
#include<stdlib.h>
#include<signal.h>
#define NMAX 3

int count_int = 0, count_quit = 0;

void handler(int sig){
    count_int++;
    if(sig == SIGINT){printf(" (%d/%d)\n" ,count_int, NMAX);
    }else if (sig == SIGQUIT){
        count_quit++;
        printf("(%d/%d)\n", count_quit, NMAX);
    }if (sig == SIGINT || sig == SIGQUIT){
        printf("Fin du programme.");
        exit(0);
    }
}
int main(){
    struct sigaction sa;
    sa.sa_handler = handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);

    while(1){pause();}

}

