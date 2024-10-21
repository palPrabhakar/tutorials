#ifndef QUEUE_H
#define QUEUE_H

#include <stddef.h>

typedef struct queue {
  void **data;
  size_t top;
  size_t end;
  size_t size;
  size_t max;
} queue_t;

// size_t: max_elements
void init_queue(queue_t *, size_t);

// queue_t: valid queue ptr
void free_queue(queue_t *);

// void *: data
// success: return 0
// failure: return 0
int queue_push(queue_t *, void *);
int queue_pop(queue_t *, void **);
size_t queue_size(queue_t *);

#endif
