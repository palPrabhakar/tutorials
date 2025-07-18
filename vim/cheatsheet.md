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

## Getting Around Faster

### Vim Motions

Motions are not just for navigation around a document. They can also be used in operator-pending mode perform real work.

| Description | Command |
|--------|---------|
| Move up and down real lines | j/k |
| Move up and down display lines | gj/gk |
| To the first character of real line | 0 |
| To the first character of display line | g0 |
| To the first non-blank character of real line | ^ |
| To the first non-blank character of display line | g^ |
| End of real line | $ |
| End of display line | g$ |

### words vs WORDS

We can think of WORDS and bigger than words.

| Description | Command (words) | Command (WORDS) |
|--------|---------|---------|
| Forward to start of next word | w | W |
| Backward to start of current/previous word | b | B|
| Forward to end of current/next word | e | E |
| Backward to end of previous word | ge | gE |

### Find by Character

The `f{char}` command is one of the quickest methods of moving around in Vim. It searches for the specified character, starting with the cursor position and continuing to the end of the current line.

| Description | Command |
|--------|---------|
| Forward to the next occurrence of {char} | f{char} |
| Backward to the previous occurrence of {char} | F{char} |
| Forward to the character before the next occurrence of {char} | t{char} |
| Backward to the character after the previous occurrence of {char} | T{char} |
| Repeat the last character-search command | ; |
| Reverse the last character-search command | , |

### Text Objects

Text objects define regions of text by structure. With only a couple of keystrokes, we can use these to select or operate on a chunk of text. Vim’s text objects consist of two characters, the first of which is always either i or a. As a mnemonic, think of i as inside and a as around (or all).

Remember this: whenever you see {motion} as part of the syntax for a command, you can also use a text object.

Vim’s text objects fall into two categories: those that interact with pairs of delimiters, such as `i)` , `i"` , and `it` , and those that interact with chunks of text, such as words `iw or iW`, sentences as `is` , and paragraphs `ap`.

### Vim Marks

Vim’s marks allow us to jump quickly to locations of interest within a document. We can set marks manually, but Vim also keeps track of certain points of interest for us automatically.

The `m{a-zA-Z}` command marks the current cursor location with the designated letter. Lowercase marks are local to each individual buffer, whereas uppercase marks are globally accessible. Vim provides two Normal mode commands for jumping to a mark. `’{mark}` moves to the line where a mark was set, positioning the cursor on the first non-whitespace character. The ```{mark}`` command moves the cursor to the exact position where a mark was set, restoring the line and the column at once.

## Jumps

Jumps are similar to motions, except that they can also move us between different files.

### Jump List

Any command that changes the active file for the current window can be described as a jump. In the jump list, Vim records the cursor location before and after running such a command.

The `<C-o>` command is like the back button, while the complementary `<C-i>` command is like the forward button. These commands allow us to traverse Vim’s jump list. We can inspect the contents of the jump list by running this command: `:jumps`.

| Description | Command |
|--------|---------|
| Jump to line number | [count]G |
| Jump to next/previous occurrence of pattern | //pattern<CR>/?pattern<CR>/n/N |
| Jump to matching parenthesis | % |
| Jump to start of previous/next sentence | (/) |
| Jump to start of previous/next paragraph | {/} |
| Jump to top/middle/bottom of screen | H/M/L |
| Jump to file name under the cursor | gf |
| Jump to definition of keyword under the cursor | <C-]> |
| Jump to a mark | ``'{mark}/`{mark}``|

### Change List

Vim records the location of our cursor after each change we make to a document. We can inspect the contents of the change list by running the command: `:changes`. Using the `g;` and `g,` commands, we can traverse backward and forward through the change list. If we leave Insert mode and then scroll around the document, we can quickly carry on where we left off by pressing `gi`.

## Registers

Vim’s registers are simply containers that hold text. They can be used in the manner of a clipboard for cutting, copying, and pasting text, or they can be used to record a macro by saving a sequence of keystrokes.

### Copy and Paste

Instead of dealing with a single system-wide clipboard, Vim provides a couple of dozen registers where we can store regions of text. We can specify which register we want to use by prefixing the command with `"{register}`.

In addition to the Normal mode commands, Vim also provides Ex commands for delete, yank, and put operations. We could cut the current line into register c by running `:delete c`, and then we could paste it below the current line with the `:put c` command.

Vim also provides `gp` and `gP` commands. These also put the text before or after the current line, but they leave the cursor positioned at the end of the pasted text instead of at the beginning.

### Unnamed Register ("")

The `x` , `s` , `d{motion}` , `c{motion}` , and `y{motion}` commands (and their uppercase equivalents) all set the contents of the unnamed register.

### Yank Register ("0)

When we use the y{motion} command, the specified text is copied not only into the unnamed register but also into the yank register, which is addressed by the 0 symbol.

### Named Registers ("a-"z)

Vim has one named register for each letter of the alphabet.

### Blackhole Register ("_)

The black hole register is a place from which nothing returns. It’s addressed by the underscore symbol.

### System Clipboard ("+) and Selection ("*) Registers

If we prefix Vim’s yank or delete commands with `"+`, the specified text will be captured in the system clipboard. The X11 windowing system has a second kind of clipboard called the primary. This represents the most recently selected text. Vim’s quotestar register maps to the primary clipboard and is addressed by the `*` symbol.

### Expression Register ("=)

Vim’s registers can be thought of simply as containers that hold a block of text. The expression register, referenced by the `=` symbol, is an exception.

## Replace Mode

Replace mode is identical to insert mode, except that it overwrites the existing text in the document. Press 'R' to engage replace mode.

## Select Mode

Select mode is similar to visual mode, however, the selected text is deleted when we type any printable character. Press '\<C-g\>' to toggle between visual mode and select mode.

## Notes

1. When an operator command is invoked in duplicate, it acts upon the current line eg - dd.
