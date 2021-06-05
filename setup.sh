mkdir -p ~/.streamlit/
echo "\
[global]\n\
showWarningOnDirectExecution = false\n\
\n\
[theme]\n\
base=\"dark\"\n\
\n\
[logger]\n\
level=\"info\"\n\
\n\
[client]\n\
caching = true\n\
\n\
[server]\n\
headless = true\n\
runOnSave = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml