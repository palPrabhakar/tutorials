#ifndef RBT_H
#define RBT_H
#include <stdio.h>

typedef enum color { red, black } color_t;

typedef struct node node_t;

typedef struct tree tree_t;

node_t *create_node(void *);

// return a < b
typedef int (*comparator)(void *, void *);
tree_t *init_tree(comparator);

// use create_node to create new node
void rb_insert(tree_t *, node_t *);
void rb_delete(tree_t *, node_t *);

// FILE *: write to file
// to_string: function to convert data to char *
//    -char *: buffer
//    -void *: data
// size_t: buffer_size
// buffer_size should be large enough to
// hold char * converted data
typedef void (*to_string)(char *, void *);
void rb_print_tree(tree_t *, FILE *, to_string, size_t);

node_t *rb_root(tree_t *);

#endif
