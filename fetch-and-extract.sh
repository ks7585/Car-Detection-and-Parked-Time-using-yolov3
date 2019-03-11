
#!/bin/bash


wget -q $1
#echo downloading $1
ffmpeg -ss 0.0 -i $2.ts -vframes 1  $2.jpg &>ffmpeg_logs.txt
#echo extracting image ... wrote  $2.jpg
rm -rf $2.ts


