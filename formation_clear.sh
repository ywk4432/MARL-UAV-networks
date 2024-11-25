#!/bin/bash

# 这是一个注释，说明了这个脚本的作用

# 指定要删除内容的文件夹路径
folder_path="/path/to/your/folder"

# 使用rm命令删除文件夹中的所有内容
rm -rf "results/sacred/formation"/*
rm -rf "record/formation"/*
rm -rf "fig/TD_error_abs/two_timescale"/*

# 输出删除完成的消息
echo "All files of Formation testing have been deleted."
