#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void *tache(void *arg) {
    int i = *(int *)arg;

    for (int j = 0; j < 10000; j++);

    printf("Thread %d -> PID: %d | TID: %lu\n",
           i, (int)getpid(), (unsigned long)pthread_self());

    // if (i == 3) exit(0); // terminates ALL threads

    return NULL;
}

int main() {
    pthread_t threads[5];
    int ids[5] = {1, 2, 3, 4, 5};

    printf("Thread principal -> PID: %d | TID: %lu\n",
           (int)getpid(), (unsigned long)pthread_self());

    for (int i = 0; i < 5; i++) {
        pthread_create(&threads[i], NULL, tache, &ids[i]);
    }

    // Avoid premature termination of main thread
    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Tous les threads sont terminés.\n");
    return 0;
}
