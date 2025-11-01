#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

void *thread_fonction(void *arg) {
    int num = *(int *)arg;
    printf("Thread fils #%d -> PID: %d | TID: %lu\n",
           num, getpid(), pthread_self());
    pthread_exit(NULL);
}

int main() {
    pthread_t threads[3];
    int args[3] = {1, 2, 3};

    printf("Thread principal -> PID: %d | TID: %lu\n",
           getpid(), pthread_self());

    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, thread_fonction, &args[i]);
    }

    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
