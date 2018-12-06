# Because I can't be bothered
clear

# Get the whole string that was given in the input
str="'$*'"
echo "$str"

# Use the string as a commit message
git add --all

# Input argument (first)
git commit -m "$str"
git push 

clear