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
queue_t *init_queue(size_t);

// void *: data
// success: return 0
// failure: return 0
int queue_push(queue_t *, void *);
int queue_pop(queue_t *, void **);

#endif
