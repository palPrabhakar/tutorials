#include <stdio.h>
#include <stdlib.h>

#include "queue.h"

struct queue {
  void **data;
  size_t top;
  size_t end;
  size_t size;
  size_t max;
};

queue_t *init_queue(size_t size) {
  queue_t *queue = malloc(sizeof(queue_t));
  queue->data = malloc(size * sizeof(void *));
  queue->max = size;
  queue->size = 0;
  queue->top = 0;
  queue->end = 0;
  return queue;
}

int queue_push(queue_t *queue, void *elem) {
  // queue full
  if (queue->size == queue->max) {
    return 1;
  }

  void **ptr = queue->data + queue->end;
  *ptr = elem;
  queue->size++;
  queue->end = (queue->end + 1) % queue->max;
  return 0;
}

int queue_pop(queue_t *queue, void **elem) {
  // queue empty
  if (queue->size == 0) {
    return 1;
  }

  void **ptr = queue->data + queue->top;
  *elem = *ptr;
  queue->size--;
  queue->top = (queue->top + 1) % queue->max;
  return 0;
}

size_t queue_size(queue_t *queue) { return queue->size; }