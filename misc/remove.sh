# #!/bin/bash

# files=(
#         "/home/prudvik/id-dataset/id-dataset/casiab/003/nm-04/072/003-nm-04-072_b55_purpleshirt_magentapant.mp4"                                             
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/cl-01/108/005-cl-01-108_b44_limeshirt_maroonpant.mp4"
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/cl-02/090/005-cl-02-090_b56_tealshirt_greenpant.mp4"   
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/072/005-nm-05-072_b53_navyshirt_redpant.mp4"     
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/090/005-nm-05-090_b26_grayshirt_olivepant.mp4"   
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/090/005-nm-05-090_b51_yellowshirt_bluepant.mp4"  
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/108/005-nm-05-108_b16_blackshirt_tealpant.mp4"   
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/108/005-nm-05-108_b17_oliveshirt_olivepant.mp4"  
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/108/005-nm-05-108_b51_greenshirt_purplepant.mp4" 
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/108/005-nm-05-108_b56_navyshirt_magentapant.mp4" 
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/126/005-nm-05-126_b16_yellowshirt_violetpant.mp4"
#         "/home/prudvik/id-dataset/id-dataset/casiab/005/nm-05/126/005-nm-05-126_b56_whiteshirt_purplepant.mp4" 
#         "/home/prudvik/id-dataset/id-dataset/casiab/006/nm-03/072/006-nm-03-072_b56_blueshirt_olivepant.mp4"   
#         "/home/prudvik/id-dataset/id-dataset/casiab/006/nm-03/108/006-nm-03-108_b61_greenshirt_purplepant.mp4" 
#         "/home/prudvik/id-dataset/id-dataset/casiab/009/nm-03/018/009-nm-03-018_b32_blueshirt_cyanpant.mp4"    
#         "/home/prudvik/id-dataset/id-dataset/casiab/021/bg-01/108/021-bg-01-108_b50_brownshirt_bluepant.mp4"  
#         "/home/prudvik/id-dataset/id-dataset/casiab/023/nm-04/054/023-nm-04-054_b10_silvershirt_orangepant.mp4"                                                                       
#         "/home/prudvik/id-dataset/id-dataset/casiab/024/cl-01/072/024-cl-01-072_b61_yellowshirt_bluepant.mp4" 
#         "/home/prudvik/id-dataset/id-dataset/casiab/024/cl-01/090/024-cl-01-090_b61_purpleshirt_magentapant.mp4"                                                                      
#         "/home/prudvik/id-dataset/id-dataset/casiab/028/bg-01/126/028-bg-01-126_b61_greenshirt_yellowpant.mp4"
#         "/home/prudvik/id-dataset/id-dataset/casiab/029/cl-02/126/029-cl-02-126_b32_blueshirt_tealpant.mp4"
#         "/home/prudvik/id-dataset/id-dataset/casiab/047/nm-04/090/047-nm-04-090_b56_maroonshirt_silverpant.mp4"                                                                       
#         "/home/prudvik/id-dataset/id-dataset/casiab/047/nm-04/108/047-nm-04-108_b56_purpleshirt_whitepant.mp4"
#         "/home/prudvik/id-dataset/id-dataset/casiab/065/nm-05/072/065-nm-05-072_b10_yellowshirt_graypant.mp4"
# )

# for file in "${files[@]}"
# do
#     if [ -f "$file" ]; then
#         rm "$file"
#         echo "Removed file: $file"
#     else
#         echo "File not found: $file"
#     fi
# done

#!/bin/bash

root_dir="/home/prudvik/id-dataset/id-dataset/casiab"

delete_directory() {
    local dir="$1"
    
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo "Deleted directory: $dir"
    fi
}

traverse_directories() {
    local dir="$1"

    for sub_dir in "$dir"/*; do
        if [ -d "$sub_dir" ]; then
            if [ "$(basename "$sub_dir")" = "bkgrd" ]; then
                delete_directory "$sub_dir"
            else
                traverse_directories "$sub_dir"
            fi
        fi
    done
}

traverse_directories "$root_dir"
