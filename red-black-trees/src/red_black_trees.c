#include <stdio.h>

#include "../../queue/src/queue.h"
#include "red_black_trees.h"

#define CHECK_QUEUE(func, ...)                                                 \
  if (func) {                                                                  \
    fprintf(stderr, __VA_ARGS__);                                              \
    break;                                                                     \
  }

static node_t nil = {.color = black};

static void rb_insert_fixup(tree_t *, node_t *);
static void rb_rotate_left(tree_t *, node_t *);
static void rb_rotate_right(tree_t *, node_t *);
static void rb_transplant(tree_t *, node_t *, node_t *);
static void rb_delete_fixup(tree_t *, node_t *);
static node_t *tree_minimum(node_t *);
static int rb_node_impl(tree_t *, node_t *, void *, node_t **);
static void traverse_tree_impl(node_t *, call_back);

static node_t *tree_minimum(node_t *node) {
  while (node->left != &nil) {
    node = node->left;
  }

  return node;
}

void init_tree(tree_t *tree, comparator f) {
  tree->root = &nil;
  tree->compare = f;
  tree->size = 0;
}

static void traverse_tree_impl(node_t *node, call_back cb) {
  if (node != &nil) {
    // node might get deleted during traversal
    node_t *right = node->right;
    traverse_tree_impl(node->left, cb);
    cb(node);
    traverse_tree_impl(right, cb);
  }
}

void traverse_tree(tree_t *tree, call_back cb) {
  traverse_tree_impl(tree->root, cb);
}

void init_node(node_t *node, void *data) {
  node->color = red;
  node->data = data;
  node->right = &nil;
  node->left = &nil;
  node->parent = &nil;
}

node_t *rb_root(tree_t *tree) { return tree->root; }

static int rb_node_impl(tree_t *tree, node_t *cur, void *key, node_t **node) {
  if (cur == &nil) {
    return 1;
  }

  int cond = tree->compare(key, cur->data);

  if (cond == 0) {
    *node = cur;
    return 0;
  } else if (cond == -1) {
    return rb_node_impl(tree, cur->left, key, node);
  } else {
    return rb_node_impl(tree, cur->right, key, node);
  }
}

int rb_node(tree_t *tree, void *key, node_t **node) {
  return rb_node_impl(tree, tree->root, key, node);
}

void rb_insert(tree_t *tree, node_t *node) {
  node_t *parent = &nil;
  node_t *child = tree->root;

  while (child != &nil) {
    parent = child;
    if (tree->compare(node->data, child->data) == -1) {
      child = child->left;
    } else {
      child = child->right;
    }
  }

  node->parent = parent;
  if (parent == &nil) {
    tree->root = node;
  } else if (tree->compare(node->data, parent->data) == -1) {
    parent->left = node;
  } else {
    parent->right = node;
  }

  rb_insert_fixup(tree, node);
  tree->size++;
}

static void rb_insert_fixup(tree_t *tree, node_t *node) {
  node_t *uncle;
  while (node->parent->color == red) {
    if (node->parent == node->parent->parent->left) {
      uncle = node->parent->parent->right;
      if (uncle->color == red) {
        node->parent->color = black;
        uncle->color = black;
        node->parent->parent->color = red;
        node = node->parent->parent;
      } else {
        if (node == node->parent->right) {
          node = node->parent;
          rb_rotate_left(tree, node);
        }

        node->parent->color = black;
        node->parent->parent->color = red;

        rb_rotate_right(tree, node->parent->parent);
      }
    } else {
      uncle = node->parent->parent->left;
      if (uncle->color == red) {
        node->parent->color = black;
        uncle->color = black;
        node->parent->parent->color = red;
        node = node->parent->parent;
      } else {
        if (node == node->parent->left) {
          node = node->parent;
          rb_rotate_right(tree, node);
        }

        node->parent->color = black;
        node->parent->parent->color = red;
        rb_rotate_left(tree, node->parent->parent);
      }
    }
  }

  tree->root->color = black;
}

static void rb_rotate_left(tree_t *tree, node_t *node) {
  node_t *child = node->right;
  node->right = child->left;
  if (child->left != &nil) {
    child->left->parent = node;
  }

  child->parent = node->parent;
  if (node->parent == &nil) {
    tree->root = child;
  } else if (node == node->parent->left) {
    node->parent->left = child;
  } else {
    node->parent->right = child;
  }

  child->left = node;
  node->parent = child;
}

static void rb_rotate_right(tree_t *tree, node_t *node) {
  node_t *child = node->left;
  node->left = child->right;
  if (child->right != &nil) {
    child->right->parent = node;
  }

  child->parent = node->parent;
  if (node->parent == &nil) {
    tree->root = child;
  } else if (node == node->parent->left) {
    node->parent->left = child;
  } else {
    node->parent->right = child;
  }

  child->right = node;
  node->parent = child;
}

static void rb_transplant(tree_t *tree, node_t *u, node_t *v) {
  if (u->parent == &nil) {
    tree->root = v;
  } else if (u == u->parent->left) {
    u->parent->left = v;
  } else {
    u->parent->right = v;
  }
  v->parent = u->parent;
}

int rb_remove_key(tree_t *tree, void *key, node_t **node) {
  if (rb_node(tree, key, node)) {
    return 1;
  }

  rb_delete(tree, *node);

  return 0;
}

void rb_delete(tree_t *tree, node_t *node) {
  node_t *tmp = node;
  color_t tmp_color = tmp->color;
  node_t *tmp_succ;

  if (node->left == &nil) {
    tmp_succ = node->right;
    rb_transplant(tree, node, node->right);
  } else if (node->right == &nil) {
    tmp_succ = node->left;
    rb_transplant(tree, node, node->left);
  } else {
    tmp = tree_minimum(node->right);
    tmp_color = tmp->color;
    tmp_succ = tmp->right;

    if (tmp->parent == node) {
      /*
      Although you might think that setting x.p to y in line 16 is unnecessary
      since x is a child of y, the call of RB-DELETE-FIXUP relies on x.p being
      y even if x is T.nil. -Cormen et. al 4th ed.
      */
      tmp_succ->parent = tmp;
    } else {
      rb_transplant(tree, tmp, tmp->right);
      tmp->right = node->right;
      tmp->right->parent = tmp;
    }

    rb_transplant(tree, node, tmp);
    tmp->left = node->left;
    tmp->left->parent = tmp;
    tmp->color = node->color;
  }

  if (tmp_color == black) {
    rb_delete_fixup(tree, tmp_succ);
  }
}

static void rb_delete_fixup(tree_t *tree, node_t *node) {
  // If node is a red-black node then skip while loop and
  // simply color the node black
  // If node is root then skip while loop and simply
  // remove the blackness
  // else:
  node_t *sibling;
  while (node != tree->root && node->color == black) {
    if (node == node->parent->left) {
      sibling = node->parent->right;

      if (sibling->color == red) {
        sibling->color = black;
        node->parent->color = red;
        rb_rotate_left(tree, node->parent);
        sibling = node->parent->right;
      }

      if (sibling->left->color == black && sibling->right->color) {
        sibling->color = red;
        node = node->parent;
      } else {
        if (sibling->right->color == black) {
          sibling->left->color = black;
          sibling->color = red;
          rb_rotate_right(tree, sibling);
          sibling = node->parent->right;
        }

        sibling->color = node->parent->color;
        node->parent->color = black;
        sibling->right->color = black;
        rb_rotate_left(tree, node->parent);
        node = tree->root;
      }
    } else {
      sibling = node->parent->left;

      if (sibling->color == red) {
        sibling->color = black;
        node->parent->color = red;
        rb_rotate_left(tree, node->parent);
        sibling = node->parent->left;
      }

      if (sibling->left->color == black && sibling->right->color == black) {
        sibling->color = red;
        node = node->parent;
      } else {
        if (sibling->left->color == black) {
          sibling->right->color = black;
          sibling->color = red;
          rb_rotate_left(tree, sibling);
          sibling = node->parent->left;
        }

        sibling->color = node->parent->color;
        node->parent->color = black;
        sibling->left->color = black;
        rb_rotate_right(tree, node->parent);
        node = tree->root;
      }
    }
  }

  node->color = black;
}

static const char *node_color(color_t color) {
  switch (color) {
  case red:
    return "red";
  case black:
    return "black";
  default:
    return "";
  }
}

void rb_print_tree(tree_t *tree, FILE *f, to_string to_str, size_t buf_size) {
  fprintf(f, "digraph RedBlackTree {\n");
  fprintf(f, "node [shape=circle style=filled];\n");

  char node_buffer[buf_size];
  char child_buffer[buf_size];

  if (tree->root != &nil) {

    queue_t queue;
    init_queue(&queue, tree->size);

    queue_push(&queue, tree->root);

    while (queue_size(&queue) != 0) {
      node_t *node;
      CHECK_QUEUE(queue_pop(&queue, (void **)&node),
                  "Error: queue_pop() failed at %d.\n", __LINE__)
      to_str(node_buffer, node->data);
      fprintf(f, "%s [fillcolor=%s fontcolor=white];\n", node_buffer,
              node_color(node->color));

      if (node->left != &nil) {
        to_str(child_buffer, node->left->data);
        fprintf(f, "%s -> %s;\n", node_buffer, child_buffer);
        CHECK_QUEUE(queue_push(&queue, node->left),
                    "Error: queue_push() failed at %d.\n", __LINE__);
      }

      if (node->right != &nil) {
        to_str(child_buffer, node->right->data);
        fprintf(f, "%s -> %s;\n", node_buffer, child_buffer);
        CHECK_QUEUE(queue_push(&queue, node->right),
                    "Error: queue_push() failed at %d.\n", __LINE__);
      }
    }

    free_queue(&queue);
  }

  fprintf(f, "}\n");
}
