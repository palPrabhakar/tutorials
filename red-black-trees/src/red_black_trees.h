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

// typedef const char * (*to_string)(void *);
// void rb_print_tree(tree_t *, to_string);
void rb_print_tree(tree_t *, FILE *);

node_t *rb_root(tree_t *);

#endif
