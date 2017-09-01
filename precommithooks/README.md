# TL;DR — pre-commit hooks can save you from embarrassment and improve workflow

## Workflow Improvements
Not everything is big data, machine or deep learning at Lab41. Sometimes I need to take a step back and look at how I’m accomplishing my goals. I feel that I should evaluate how I do things more often. I have noticed my co-workers and I set large goals and short timelines. Sometimes rushed, I cut a review corner, and commit things too quickly.  

I’m writing this as part of my journey of not committing internal IP addresses, private naming conventions, or using embarrassing words to a git repository. I have found, even under the best conditions, I will accidentally commit something I regret to a repo at least once a project.  

On the bottom of my monitor I keep a tally of the repo resets that I have had to perform since starting at the lab. Even with the visual reminder I still manage to pollute the repo. I am baking compliance into the check-in process with hope that you will be able to improve your development environment in a similar way.  

## Deciding on pre-commit
Git can run custom scripts as a user transitions within their development workflow. There are local and server-side hooks for making decisions based on file contents and actions performed. If a hook test fails, then the transaction is aborted and the repo does not get updated.  

I would like to walk through my use of a pre-commit script I use to keep ‘bad’ words out of my commits. After reading an excellent summary and tutorial by Atlassian:[Git Hook Tutorial] I decided that implementing a local hook would suit my needs best. I could experiment in my local repo and not bother the other developers in my team.  

Recently most of my development is done in python, but I still work in other languages. I decided to install Yelp’s pre-commit as it has a supportive community. Yelp’s pre-commit is a framework for managing multi-language pre-commit hooks. The framework brokers running my custom hook along with community hooks, improving my code quality and readability.  

## Installation and community hooks
Before you can take advantage of running the hooks, you need to install the package.

`pip install pre-commit`  

There are a slew of existing checks that can be immediately added and utilized. I looked through the listing and found several that I thought would make future dgrossman’s life less miserable. (Future dgrossman’s life is seldom made better by past or present dgrossman’s decisions.)  

>autopep8-wrapper - Runs autopep8 over python source.  
>check-case-conflict - Check for files that would conflict in case-insensitive filesystems.  
>check-json - This hook checks json files for parseable syntax.  
>pretty-format-json - This hook sets a standard for formatting JSON files.  
>check-merge-conflict - Check for files that contain merge conflict strings.  
>check-symlinks - Checks for symlinks which do not point to anything.  
>check-yaml - This hook checks yaml files for parseable syntax.  
>end-of-file-fixer - Ensures that a file is either empty, or ends with one newline.  
>trailing-whitespace - This hook trims trailing whitespace.  


After picking the pre-existing modules I needed to make a .pre-commit-config.yaml file in the root of my repo:
```
-   repo: https://github.com/pre-commit/pre-commit
    sha: v0.9.4
    hooks:
    -   id: validate_config
-   repo: git@github.com:pre-commit/pre-commit-hooks
    sha: v0.6.1
    hooks:
    -   id: autopep8-wrapper
    -   id: check-case-conflict
    -   id: check-json
    -   id: check-merge-conflict
    -   id: check-symlinks
    -   id: check-yaml
    -   id: end-of-file-fixer
    -   id: pretty-format-json
        args:
        - --autofix
    -   id: trailing-whitespace
-   repo: git@github.com:asottile/reorder_python_imports
    sha: v0.3.0
    hooks:
    -   id: reorder-python-imports
```  

Finally, I was ready to perform the installation and verify the correctness of the .pre-commit-config.yaml file. Running pre-commit in the repo started the installation.  

`~/work/myproj$ pre-commit`
>[INFO] Initializing environment for git@github.com:pre-commit/pre-commit-hooks.  
>[INFO] Installing environment for git@github.com:pre-commit/pre-commit-hooks.  
>[INFO] Once installed this environment will be reused.  
>[INFO] This may take a few minutes…  
>Validate Pre-Commit Config………………………(no files to check)Skipped  
>autopep8 wrapper……………………………….(no files to check)Skipped  
>Check for case conflicts………………………..(no files to check)Skipped  
>Check JSON…………………………………….(no files to check)Skipped  
>Check for merge conflicts……………………….(no files to check)Skipped  
>Check for broken symlinks……………………….(no files to check)Skipped  
>Check Yaml…………………………………….(no files to check)Skipped  
>Fix End of Files……………………………….(no files to check)Skipped  
>Pretty format JSON……………………………..(no files to check)Skipped  
>Trim Trailing Whitespace………………………..(no files to check)Skipped  
>Reorder python imports………………………….(no files to check)Skipped  

Now all the files I attempt to commit will need to pass before they can make it to the repo. Perfect time to add my hook for stopping bad words from making it to the repo.  

## How to write a pre-commit hook that installs from a repo

I used Yelp’s guidelines in making the commit hook:

1. choose language that pre-commit supports
1. make sure project is an installable package
1. make sure the hook exits non-zero on failure
1. make sure the hook takes filenames as positional arguments

You will need to make a minimum of 4 files to make a python pre-commit hook.  

1. setup.py -> standard package setup file
1. hooks.yaml -> yaml file describing your hook, and the types of files you expect to feed it
1. your_package/your_test.py -> The program that will be evaluating each file
1. your_package/__init__.py -> standard python init

Pre-commit hooks receive a listing of filenames for evaluation. How you evaluate the files is up to the individual author. The verboten_words repo serves as a good starting point, but the basic sketch of what to do for each file is:  

```pyton
open badwordFile named in environment variable
read bad words from badwordFile
if bad word in file:
    return HORROR
else:
    return KITTEN_PICTURE
```

After writing up the program and testing it, I tagged the version v1.0.0 and then added it to my .pre-commit-config.yaml :  


```
-   repo: https://github.com/pre-commit/pre-commit
    sha: v0.9.4
    hooks:
    -   id: validate_config
-   repo: git@github.com:pre-commit/pre-commit-hooks
    sha: v0.6.1
    hooks:
    -   id: autopep8-wrapper
    -   id: check-case-conflict
    -   id: check-json
    -   id: check-merge-conflict
    -   id: check-symlinks
    -   id: check-yaml
    -   id: end-of-file-fixer
    -   id: pretty-format-json
        args:
        - --autofix
    -   id: trailing-whitespace
-   repo: git@github.com:asottile/reorder_python_imports
    sha: v0.3.0
    hooks:
    -   id: reorder-python-imports
-   repo: git@github.com:Lab41/verboten_words.git
    sha: v1.0.0
    hooks:
    -   id: verboten-words
```

I then re-run pre-commit to install the new hook:  

>~/work/myproj$ pre-commit  
>Validate Pre-Commit Config………………………(no files to check)Skipped  
>autopep8 wrapper……………………………….(no files to check)Skipped  
>Check for case conflicts………………………..(no files to check)Skipped  
>Check JSON…………………………………….(no files to check)Skipped  
>Check for merge conflicts……………………….(no files to check)Skipped  
>Check for broken symlinks……………………….(no files to check)Skipped  
>Check Yaml…………………………………….(no files to check)Skipped  
>Fix End of Files……………………………….(no files to check)Skipped  
>Pretty format JSON……………………………..(no files to check)Skipped  
>Trim Trailing Whitespace………………………..(no files to check)Skipped  
>Reorder python imports………………………….(no files to check)Skipped  
>verboten words…………………………………(no files to check)Skipped  
>My module for not allowing bad words is now installed.  

Now my bad word file can be used to keep specific words out of my repo, or be extended to handle more advanced searches for regular expressions that are also deemed naughty.  

To get a quick start on your own hook, clone/check out the verboten_words repo on GitHubi [https://github.com/Lab41/verboten_words]. 

## Closing Thoughts
* Git hooks are a low cost way to customize your workflow
* Git hooks make it easier to automate doing the right thing with your repository; bake your compliance into the workflow.

dgrossman
