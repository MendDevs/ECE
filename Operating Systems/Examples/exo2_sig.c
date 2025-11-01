#include<stdio.h>
#include<signal.h>

void handler_self(int sig){
    printf("Send signal to self: %d",sig);
    exit(0);
}

int main(){
        signal(SIGTERM, handler_self);
        printf("Self terminate...\n");
        raise(SIGTERM);
        return 0;
}