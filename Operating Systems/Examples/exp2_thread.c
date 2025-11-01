#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  // for sleep()
#include <pthread.h> // for pthread_create, pthread_join

int glob = 0;
pthread_t tid1, tid2;

void *increment(void *arg)
{
    int inc = 2;
    for (int i = 0; i < 5; i++)
    {
        glob += inc;
        printf("ici increment [%lu], glob = %d\n", pthread_self(), glob);
        sleep(1);
    }
    pthread_exit(NULL);
}

void *decrement(void *arg)
{
    int dec = 1;
    for (int i = 0; i < 5; i++)
    {
        glob -= dec;
        printf("ici decrement [%lu], glob = %d\n", pthread_self(), glob);
        sleep(1);
    }
    pthread_exit(NULL);
}

int main()
{
    printf("ici main [%d], glob = %d\n", getpid(), glob);

    // création d’un thread pour increment
    if (pthread_create(&tid1, NULL, increment, NULL) != 0)
    {
        perror("Erreur creation thread increment");
        return 1;
    }
    printf("ici main : création du thread [%lu] increment avec succès\n", tid1);

    // création d’un thread pour decrement
    if (pthread_create(&tid2, NULL, decrement, NULL) != 0)
    {
        perror("Erreur creation thread decrement");
        return 1;
    }
    printf("ici main : création du thread [%lu] decrement avec succès\n", tid2);

    // attendre la fin des threads
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    printf("ici main : fin des threads, glob = %d\n", glob);
    return 0;
}
