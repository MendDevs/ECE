#include<stdio.h>
#include<unistd.h>
#include<pthread.h>

void *fonction(void *arg){
    printf("Thread fils-> PID: %d | TID: %lu ",
           (int)getpid(),
           (unsigned long)pthread_self()
        );
    }

int main(){
    pthread_t thread;
    printf("Thread principal -> PID: %d | TID: %lu\n",
            (int)getpid(),
            (unsigned long)pthread_self
    );

    //create a new thread
    pthread_create(&thread, NULL, &fonction, NULL);

    //wait fo this thread to finish
    pthread_join(fonction,NULL);
    return 0;
}