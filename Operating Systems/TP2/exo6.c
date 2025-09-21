#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s prog1 prog2 ...\n", argv[0]);
        exit(1);
    }

    int fails = 0;

    for (int i = 1; i < argc; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            execlp(argv[i], argv[i], NULL);
            perror("exec failed");
            exit(1);
        } else {
            int status;
            waitpid(pid, &status, 0);
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                fails++;
                break; // stop car un fichier a échoué
            }
        }
    }

    printf("Nombre d'échecs: %d\n", fails);
    return 0;
}

