#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define MAX 10
pthread_mutex_t chopsticks[MAX];
int N;

void* philosopher(void* arg) {
    int id = *(int*)arg;
    int left = id;
    int right = (id+1) % N;

    while(1) {
        printf("Philosopher %d is thinking\n", id);
        sleep(rand()%3 + 1);

        // pick up chopsticks (avoid deadlock)
        if(left < right) {
            pthread_mutex_lock(&chopsticks[left]);
            pthread_mutex_lock(&chopsticks[right]);
        } else {
            pthread_mutex_lock(&chopsticks[right]);
            pthread_mutex_lock(&chopsticks[left]);
        }

        printf("Philosopher %d is eating\n", id);
        sleep(rand()%2 + 1);

        pthread_mutex_unlock(&chopsticks[left]);
        pthread_mutex_unlock(&chopsticks[right]);
    }
}

int main() {
    printf("Enter number of philosophers (<=10): ");
    scanf("%d", &N);

    pthread_t threads[MAX];
    int ids[MAX];

    for(int i=0;i<N;i++)
        pthread_mutex_init(&chopsticks[i], NULL);

    for(int i=0;i<N;i++) {
        ids[i]=i;
        pthread_create(&threads[i], NULL, philosopher, &ids[i]);
    }

    for(int i=0;i<N;i++)
        pthread_join(threads[i], NULL);

    return 0;
}
