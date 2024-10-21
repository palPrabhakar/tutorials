#include "../src/queue.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT_SUCCESS_PUSH(queue, node)                                       \
  assert(!queue_push(queue, (node)) && "error: push failed.\n");

#define ASSERT_FAILURE_PUSH(queue, node)                                       \
  assert(queue_push(queue, (node)) && "error: push succeeded.\n");

#define ASSERT_SUCCESS_POP(queue, node)                                        \
  assert(!queue_pop(queue, (node)) && "error: pop failed.\n");

#define ASSERT_FAILURE_POP(queue, node)                                        \
  assert(queue_pop(queue, (node)) && "error: pop succeeded.\n");

#define ASSERT_SIZE(queue, x)                                                  \
  assert(queue_size(queue) == (x) && "error: unexpected queue size.\n");

#define ASSERT_VALUE(ptr, x)                                                   \
  assert((*ptr) == (x) && "error: unexpected value!.\n")

static void test_queue(queue_t *queue) {
  int i1 = 1;
  ASSERT_SUCCESS_PUSH(queue, &i1)
  ASSERT_SIZE(queue, 1)

  int i2 = 2;
  ASSERT_SUCCESS_PUSH(queue, &i2);
  ASSERT_SIZE(queue, 2);

  int i3 = 3;
  ASSERT_SUCCESS_PUSH(queue, &i3);
  ASSERT_SIZE(queue, 3);

  int i4 = 4;
  ASSERT_SUCCESS_PUSH(queue, &i4)
  ASSERT_SIZE(queue, 4);

  int i5 = 5;
  ASSERT_FAILURE_PUSH(queue, &i5)
  ASSERT_SIZE(queue, 4);

  int *ptr;
  ASSERT_SUCCESS_POP(queue, (void **)&ptr);
  ASSERT_SIZE(queue, 3);
  ASSERT_VALUE(ptr, 1);

  ASSERT_SUCCESS_PUSH(queue, &i5)
  ASSERT_SIZE(queue, 4);

  ASSERT_SUCCESS_POP(queue, (void **)&ptr);
  ASSERT_SIZE(queue, 3);
  ASSERT_VALUE(ptr, 2);

  ASSERT_SUCCESS_POP(queue, (void **)&ptr);
  ASSERT_SIZE(queue, 2);
  ASSERT_VALUE(ptr, 3);

  ASSERT_SUCCESS_POP(queue, (void **)&ptr);
  ASSERT_SIZE(queue, 1);
  ASSERT_VALUE(ptr, 4);

  ASSERT_SUCCESS_POP(queue, (void **)&ptr);
  ASSERT_SIZE(queue, 0);
  ASSERT_VALUE(ptr, 5);

  ASSERT_FAILURE_POP(queue, (void **)&ptr);
  ASSERT_SIZE(queue, 0);
}

int main() {
  printf("Running tests.\n");
  queue_t queue;
  init_queue(&queue, 4);
  test_queue(&queue);
  free_queue(&queue);
  printf("Test succeeded.\n");
}
