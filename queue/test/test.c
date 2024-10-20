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

static void test_queue() {
  queue_t *queue = init_queue(4);

  int *iptr = malloc(sizeof(int));
  *iptr = 1;
  ASSERT_SUCCESS_PUSH(queue, iptr)
  ASSERT_SIZE(queue, 1)

  iptr = malloc(sizeof(int));
  *iptr = 2;
  ASSERT_SUCCESS_PUSH(queue, iptr)
  ASSERT_SIZE(queue, 2)

  iptr = malloc(sizeof(int));
  *iptr = 3;
  ASSERT_SUCCESS_PUSH(queue, iptr)
  ASSERT_SIZE(queue, 3);

  iptr = malloc(sizeof(int));
  *iptr = 4;
  ASSERT_SUCCESS_PUSH(queue, iptr)
  ASSERT_SIZE(queue, 4);

  iptr = malloc(sizeof(int));
  *iptr = 5;
  ASSERT_FAILURE_PUSH(queue, iptr)
  ASSERT_SIZE(queue, 4);

  ASSERT_SUCCESS_POP(queue, (void **)&iptr);
  ASSERT_SIZE(queue, 3);
  ASSERT_VALUE(iptr, 1);

  *iptr = 5;
  ASSERT_SUCCESS_PUSH(queue, iptr)
  ASSERT_SIZE(queue, 4);

  ASSERT_SUCCESS_POP(queue, (void **)&iptr);
  ASSERT_SIZE(queue, 3);
  ASSERT_VALUE(iptr, 2);

  ASSERT_SUCCESS_POP(queue, (void **)&iptr);
  ASSERT_SIZE(queue, 2);
  ASSERT_VALUE(iptr, 3);

  ASSERT_SUCCESS_POP(queue, (void **)&iptr);
  ASSERT_SIZE(queue, 1);
  ASSERT_VALUE(iptr, 4);

  ASSERT_SUCCESS_POP(queue, (void **)&iptr);
  ASSERT_SIZE(queue, 0);
  ASSERT_VALUE(iptr, 5);

  ASSERT_FAILURE_POP(queue, (void **)&iptr);
  ASSERT_SIZE(queue, 0);
}

int main() {
  printf("Running tests.\n");
  test_queue();
  printf("Test succeeded.\n");
}
