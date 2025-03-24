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

| Description | Action | Cmd Mode |
|-------------|--------|---------|
| Delete back one char | \<C-h\> | Y |
| Delete back one word | \<C-w\> | Y |
| Delete back to start of line | \<C-u\> | Y |
| Switch to Insert Normal mode | \<C-o\> | N |
| Paste just yanked text at cursor position | \<C-r\>0 | Y |
| Paste text from register at cursor position | \<C-r\>{register} | Y |
| Access expression register | \<C-r\>= | N |
| Insert a char by its numeric code | \<C-v\>{code} or  \<C-v\>u{hexcode}| Y |

The first three commands can also be used in bash shell. The command which can also be used in cmd mode are marked 'Y'.

Expression register can be used to evaluate a piece of vim script code. Vim inserts the results from the expression register at the current cursor position.

## Visual Mode

| Effect | Command |
|--------|---------|
| Enable char-wise visual mode | v |
| Enable line-wise visual mode | V |
| Enable block-wise visual mode | \<C-v\>  |
| Reselect the last visual selection | gv |
| Toggle the free end | o |

## Command-Line Mode

Command line modes prompts us to enter an Ex command, a search pattern, or an expression. Command mode can be enabled by pressing ':' key. Pressing '/' brings up the search prompt and expression register can be accessed using \<C-r\>=.

| Effect | Command |
|--------|---------|
| Delete specified lines [into register x] | :[range]delete [x] |
| Yank specified lines [into register x] | :[range]yank [x] |
| Put the text form register x after the specified line | :[range]put [x] |
| Copy the specified lines to below the line specified by {address} | :[range]copy {address} |
| Move the specified lines to below the line specified by {address} | :[range]move {address} |
| Join the specified lines | :[range]join |
| Execute Normal mode {commands} on each specified line | :[range]normal {commands} |
| Replace occurrences of {pattern} with {string} on each specified line | :[range]substitute/{pattern}/{string}/[flags] |
| Execute the Ex command [cmd] on all specified lines where the {pattern} matches | :[range]global/{pattern}/[cmd] |

Vim has an Ex command for just about everything (see `:h ex-cmd-index`). The start and end of a range (`:{start},{end}`) can be specified using a line number, mark, or a pattern.

#### Specify a range by pattern

```
:/{pattern}/,/{pattern}/[cmd]
```

#### Modify an address using an offset

```
:{address}+n
```

#### Repeat last Ex command

To repeat a Ex command first time use `@:`. Afterwards, use `@@` to subsequently repeat it.

#### Miscellaneous

The !{motion} operator command drops us into Command-Line mode and prepopulates the [range] with the lines covered by the specified {motion}

### Command-Line Window

| Action | Command |
|--------|---------|
| Open the command-line window with history of searches | q/ |
| Open the command-line window with history of Ex commands | q: |
| Switch from command-line mode to command-line window | \<C-f\> |


Press `q:` to bring up the command-line window.

## Files

Buffers - In memory representation of a file in Vim is called a buffer.

| Action | Command |
|--------|---------|
| Write the contents of a buffer to a file | :w[rite] |
| Write the contents of a buffer to a file only if modified | :up[date] |
| Save the current buffer under the name and update buffer name | :sav[eas] {file} |
| Goto next buffer | :bn[ext] |
| Goto prev buffer | :bp[rev] |
| Show all buffers | :ls or :buffers |
| Execute an Ex command on all the buffers listed by :ls | :bufdo |
| Delete buffers | :bd[elete] N1 N2 ... or :N,M bd[elete] |
| Open a file relative to active file directory | :e[dit] %:h{filename} |

### Argument List

An arguments list is easily managed and can be useful for grouping together a collection of files for easy navigation.

| Action | Command |
|--------|---------|
| List the contents of an arg list | :args |
| Update the arguments in arg list | :args {arguments} |
| Execute an Ex command on all items in the arg list | :argdo |
| Goto next element in the arglist | :n[ext] |
| Goto next element in the arglist | :p[rev] |

Examples:

1. Populate argument list from shell cmd - ``:args `cat {file}` ``
2. Populate arguments list using globs -  `:args **/*.c`

### Split Windows

| Action | Command |
|--------|---------|
| Split the window horizontally | \<C-w\>s or :split |
| Split the window vertically | \<C-w\>v or :vsplit |
| Cycle between open windows | \<C-w\>w or \<C-w\>\<C-w\> |
| Focus the window to the left | \<C-w\>h |
| Focus the window to the right | \<C-w\>l |
| Focus the window to the above | \<C-w\>j |
| Focus the window to the below | \<C-w\>k |
| Close the active window | \<C-w\>c or :cl[ose] |
| Keep the active window, close all | \<C-w\>o or :on[ly] |

### Tabs

Tabs can be used to organize the split windows into a collection of workspaces.

| Action | Command |
|--------|---------|
| Open {filename} in a new tab | :tabe[dit] {filename} |
| Move the current windows into its own tab | \<C-w\>T |
| Close the current tab and all its windows | :tabc[lose] |
| Keep the active tab page, closing all others | :tabo[nly] |
| Goto next tab | :tabn[ext] or gt |
| Goto prev tab | :tabp[rev] or gT |

### Find Command

The `:find` command allows us to open a file by its name without having to provide a fully qualified path. To use the find command we first need to set the path.

The command `:path+=<dir>/**` will set the path such that vim will search all directories under `<dir>` when using find command.

### Netrw

| Action | Command |
|--------|---------|
| Open file explorer for current working directory | :e[dit] . |
| Open file explorer for the directory of the active buffer | :E[xplore] |
| Open file explorer for the directory of the active buffer in horizontal split | :Sexplore |
| Open file explorer for the directory of the active buffer in vertical split | :Vexplore |

## Replace Mode

Replace mode is identical to insert mode, except that it overwrites the existing text in the document. Press 'R' to engage replace mode.

## Select Mode

Select mode is similar to visual mode, however, the selected text is deleted when we type any printable character. Press '\<C-g\>' to toggle between visual mode and select mode.

## Notes

1. When an operator command is invoked in duplicate, it acts upon the current line eg - dd.
