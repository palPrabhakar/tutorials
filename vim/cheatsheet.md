## Repeatable actions and how to reverse them

| Description | Action | Repeat | Reverse |
|-------------|--------|--------|---------|
| Change | {edit} | . | u |
| Go to next char | f{char}/t{char} | ; | , |
| Go to prev char | F{char}/T{char} | ; | , |
| Search pattern for next match | /pattern<CR> | n | N |
| Search pattern for prev match | ?pattern<CR> | n | N |
| Perform substitution | :s/pattern/replacement | & | u |
| Execute a sequence of change | qx{changes}q | @x | u |

## Helpful Actions

| Description | Action |
|-------------|--------|
| Select the word under cursor | * |
| Increase indentation from the current line to end of file | >G |
| Delete till end of line and change to insert mode | C or c$ |
| Delete entire line and change to insert mode | S or ^c |
| Go to the end of line and change to insert mode | A or $a |
| Go to start of line and change to insert mode | I or ^i |
| Delete a word irrespective of cursor position | daw |
| Increment a number | \<C-a\> or n\<C-a\> |
| Decrement a number | \<C-x\> or n\<C-x\> |
