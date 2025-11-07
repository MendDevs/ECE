#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>

#define SPARSE_POSITION 10000

// Part 1: Create sparse file demonstration
int create_sparse_file_demo(const char* filename) {
    int fd;
    off_t position;
    ssize_t bytes_written;
    struct stat file_stat;
    
    printf("=== PART 1: Creating Sparse File ===\n");
    
    // Create an empty file
    fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd == -1) {
        perror("Error creating file");
        return -1;
    }
    
    printf("1. Created empty file: %s\n", filename);
    
    // Position the read/write head at the 10,000th byte
    position = lseek(fd, SPARSE_POSITION - 1, SEEK_SET);  // -1 because we count from 0
    if (position == -1) {
        perror("Error positioning file pointer");
        close(fd);
        return -1;
    }
    
    printf("2. Positioned file pointer at byte %ld\n", (long)position + 1);
    
    // Write a character at this position
    bytes_written = write(fd, "X", 1);
    if (bytes_written != 1) {
        perror("Error writing character");
        close(fd);
        return -1;
    }
    
    printf("3. Wrote character 'X' at position %ld\n", (long)position + 1);
    
    // Get file statistics
    if (fstat(fd, &file_stat) == -1) {
        perror("Error getting file statistics");
        close(fd);
        return -1;
    }
    
    printf("4. File size: %ld bytes\n", (long)file_stat.st_size);
    printf("5. Blocks allocated: %ld (each block = %ld bytes)\n", 
           (long)file_stat.st_blocks, (long)file_stat.st_blksize);
    printf("6. Actual disk usage: %ld bytes\n", 
           (long)file_stat.st_blocks * 512);  // st_blocks is in 512-byte units
    
    printf("\nAnalysis:\n");
    printf("- Expected file size: %d bytes\n", SPARSE_POSITION);
    printf("- This creates a 'sparse file' - the gap of %d bytes is not physically allocated\n", 
           SPARSE_POSITION - 1);
    printf("- Only the blocks containing actual data are allocated on disk\n");
    
    close(fd);
    return 0;
}

// Part 2: Demonstrate current offset tracking
int offset_tracking_demo(const char* filename) {
    int fd;
    off_t current_offset;
    char buffer[100];
    
    printf("\n=== PART 2: Offset Tracking Demo ===\n");
    
    // Open file for reading and writing
    fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        return -1;
    }
    
    // Get initial offset
    current_offset = lseek(fd, 0, SEEK_CUR);
    printf("1. Initial offset: %ld\n", (long)current_offset);
    
    // Read some bytes
    printf("2. Reading 5 bytes from beginning...\n");
    lseek(fd, 0, SEEK_SET);
    read(fd, buffer, 5);
    current_offset = lseek(fd, 0, SEEK_CUR);
    printf("   Offset after reading 5 bytes: %ld\n", (long)current_offset);
    
    // Write some bytes
    printf("3. Writing 3 bytes at current position...\n");
    write(fd, "ABC", 3);
    current_offset = lseek(fd, 0, SEEK_CUR);
    printf("   Offset after writing 3 bytes: %ld\n", (long)current_offset);
    
    // Seek to specific position
    printf("4. Seeking to position 100...\n");
    current_offset = lseek(fd, 100, SEEK_SET);
    printf("   New offset: %ld\n", (long)current_offset);
    
    // Seek relative to current position
    printf("5. Seeking 50 bytes forward from current position...\n");
    current_offset = lseek(fd, 50, SEEK_CUR);
    printf("   New offset: %ld\n", (long)current_offset);
    
    // Seek to end of file
    printf("6. Seeking to end of file...\n");
    current_offset = lseek(fd, 0, SEEK_END);
    printf("   Offset at end of file: %ld\n", (long)current_offset);
    
    close(fd);
    return 0;
}

// Part 3: Demonstrate shared offset between descriptors
int shared_offset_demo(const char* filename) {
    int fd1, fd2;
    off_t offset1, offset2;
    char buffer[10];
    
    printf("\n=== PART 3: Shared Offset Between Descriptors ===\n");
    
    // Open the same file with two different descriptors
    fd1 = open(filename, O_RDONLY);
    fd2 = open(filename, O_RDONLY);
    
    if (fd1 == -1 || fd2 == -1) {
        perror("Error opening file");
        if (fd1 != -1) close(fd1);
        if (fd2 != -1) close(fd2);
        return -1;
    }
    
    printf("1. Opened same file with two descriptors (fd1=%d, fd2=%d)\n", fd1, fd2);
    
    // Check initial offsets
    offset1 = lseek(fd1, 0, SEEK_CUR);
    offset2 = lseek(fd2, 0, SEEK_CUR);
    printf("2. Initial offsets - fd1: %ld, fd2: %ld\n", (long)offset1, (long)offset2);
    
    // Modify offset via fd1
    printf("3. Moving fd1 to position 50...\n");
    lseek(fd1, 50, SEEK_SET);
    offset1 = lseek(fd1, 0, SEEK_CUR);
    offset2 = lseek(fd2, 0, SEEK_CUR);
    printf("   After lseek on fd1 - fd1: %ld, fd2: %ld\n", (long)offset1, (long)offset2);
    
    // Read via fd2
    printf("4. Reading 5 bytes via fd2...\n");
    read(fd2, buffer, 5);
    offset1 = lseek(fd1, 0, SEEK_CUR);
    offset2 = lseek(fd2, 0, SEEK_CUR);
    printf("   After read on fd2 - fd1: %ld, fd2: %ld\n", (long)offset1, (long)offset2);
    
    printf("\nConclusion:\n");
    printf("- Each file descriptor has its own independent offset\n");
    printf("- Operations on one descriptor don't affect the other descriptor's offset\n");
    printf("- This is because each open() call creates a separate file description\n");
    
    close(fd1);
    close(fd2);
    return 0;
}

// Demonstration with dup() - shared offset
int dup_shared_offset_demo(const char* filename) {
    int fd1, fd2;
    off_t offset1, offset2;
    char buffer[10];
    
    printf("\n=== PART 4: Shared Offset with dup() ===\n");
    
    // Open file and duplicate descriptor
    fd1 = open(filename, O_RDONLY);
    if (fd1 == -1) {
        perror("Error opening file");
        return -1;
    }
    
    fd2 = dup(fd1);  // This creates a duplicate that shares the same file description
    if (fd2 == -1) {
        perror("Error duplicating descriptor");
        close(fd1);
        return -1;
    }
    
    printf("1. Original descriptor fd1=%d, duplicated descriptor fd2=%d\n", fd1, fd2);
    
    // Check initial offsets
    offset1 = lseek(fd1, 0, SEEK_CUR);
    offset2 = lseek(fd2, 0, SEEK_CUR);
    printf("2. Initial offsets - fd1: %ld, fd2: %ld\n", (long)offset1, (long)offset2);
    
    // Modify offset via fd1
    printf("3. Moving fd1 to position 25...\n");
    lseek(fd1, 25, SEEK_SET);
    offset1 = lseek(fd1, 0, SEEK_CUR);
    offset2 = lseek(fd2, 0, SEEK_CUR);
    printf("   After lseek on fd1 - fd1: %ld, fd2: %ld\n", (long)offset1, (long)offset2);
    
    // Read via fd2
    printf("4. Reading 3 bytes via fd2...\n");
    read(fd2, buffer, 3);
    offset1 = lseek(fd1, 0, SEEK_CUR);
    offset2 = lseek(fd2, 0, SEEK_CUR);
    printf("   After read on fd2 - fd1: %ld, fd2: %ld\n", (long)offset1, (long)offset2);
    
    printf("\nConclusion:\n");
    printf("- dup() creates descriptors that share the same file description\n");
    printf("- Operations on one descriptor affect the other's offset\n");
    printf("- This demonstrates that offset is associated with the file description, not the descriptor\n");
    
    close(fd1);
    close(fd2);
    return 0;
}

int main(int argc, char* argv[]) {
    const char* test_filename = "sparse_test_file.txt";
    
    if (argc > 1) {
        test_filename = argv[1];
    }
    
    printf("File positioning and lseek() demonstration\n");
    printf("Test file: %s\n\n", test_filename);
    
    // Remove existing file
    unlink(test_filename);
    
    // Run demonstrations
    if (create_sparse_file_demo(test_filename) != 0) {
        return 1;
    }
    
    if (offset_tracking_demo(test_filename) != 0) {
        return 1;
    }
    
    if (shared_offset_demo(test_filename) != 0) {
        return 1;
    }
    
    if (dup_shared_offset_demo(test_filename) != 0) {
        return 1;
    }
    
    printf("\n=== Additional Commands to Try ===\n");
    printf("1. Check file size: ls -l %s\n", test_filename);
    printf("2. Check disk usage: du -h %s\n", test_filename);
    printf("3. Check filesystem: df -h .\n");
    printf("4. View file content: hexdump -C %s\n", test_filename);
    
    return 0;
}
