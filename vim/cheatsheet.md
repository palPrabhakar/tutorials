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

## Actions

action = operator + motion

## Helpful Actions - Normal Mode

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
| Make uppercase | gU{motion} |
| Make lowercase | gu{motion} |
| Toggle case | g~{motion} |
| Look up the man page for the word under cursor | K |
| Join the current and next line together | J |

## Helpful Actions - Insert Mode

| Description | Action |
|-------------|--------|
| Delete back one char | \<C-h\> |
| Delete back one word | \<C-w\> |
| Delete back to start of line | \<C-u\> |
| Switch to Insert Normal mode | \<C-o\> |
| Paste just yanked text at cursor position | \<C-r\>0 |
| Paste text from register at cursor position | \<C-r\>{register} |
| Access expression register | \<C-r\>= |
| Insert a char by its numeric code | \<C-v\>{code} or  \<C-v\>u{hexcode}|

The first three commands can also be used in vim's command line and bash shell

Expression register can be used to evaluate a piece of vim script code. Vim inserts the results from the expression register at the current cursor position.

## Visual Mode

| Effect | Command |
|--------|---------|
| Enable char-wise visual mode | v |
| Enable line-wise visual mode | V |
| Enable block-wise visual mode | \<C-v\>  |
| Reselect the last visual selection | gv |
| Toggle the free end | o |

### Replace Mode

Replace mode is identical to insert mode, except that it overwrites the existing text in the document. Press 'R' to engage replace mode.

## Select Mode

Select mode is similar to visual mode, however, the selected text is deleted when we type any printable character. Press '\<C-g\>' to toggle between visual mode and select mode.

## Notes

1. When an operator command is invoked in duplicate, it acts upon the current line eg - dd.
