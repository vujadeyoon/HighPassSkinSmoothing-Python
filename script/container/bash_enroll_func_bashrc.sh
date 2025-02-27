#!/bin/bash
#
#
# Convert CSV to ODS
echo -e "\n\n## csv2ods: Convert CSV to ODS." | tee -a ~/.bashrc
echo "function csv2ods() {" | tee -a ~/.bashrc
echo "    soffice --convert-to ods \$1" | tee -a ~/.bashrc
echo "}" | tee -a ~/.bashrc
#
#
# Display the name of the git branch.
echo -e "\n\n## Display the name of the git branch." | tee -a ~/.bashrc
echo "parse_git_branch() {" | tee -a ~/.bashrc
echo "    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'" | tee -a ~/.bashrc
echo "}" | tee -a ~/.bashrc
echo "export PS1=\"\u@\h \[\033[32m\]\w\[\033[33m\]\\\$(parse_git_branch)\[\033[00m\] $ \"" | tee -a ~/.bashrc
#
#
# vid2frm: Video to frames.
echo -e "\n\n## vid2frm: Video to frames." | tee -a ~/.bashrc
echo "function vid2frm() {" | tee -a ~/.bashrc
echo "  case \$1 in" | tee -a ~/.bashrc
echo "    -h|--help)" | tee -a ~/.bashrc
echo "    echo \"Usage: vid2frm name_video.mp4\"" | tee -a ~/.bashrc
echo "    ;;" | tee -a ~/.bashrc
echo "    *)" | tee -a ~/.bashrc
echo "    path_file=\$1" | tee -a ~/.bashrc
echo "    filename_extension=\"\${path_file##*/}\" # \"\$(basename -- \$path_file)\"" | tee -a ~/.bashrc
echo "    filename=\${filename_extension%.*}" | tee -a ~/.bashrc
echo "    extension=\".\${path_file##*.}\"" | tee -a ~/.bashrc
echo "    rm -rf ./\$filename" | tee -a ~/.bashrc
echo "    mkdir -p ./\$filename" | tee -a ~/.bashrc
echo "    ffmpeg -i \$1 -start_number 0 ./\$filename/frm_%08d.png" | tee -a ~/.bashrc
echo "    ;;" | tee -a ~/.bashrc
echo "  esac" | tee -a ~/.bashrc
echo "}" | tee -a ~/.bashrc
#
#
# img2png: Image to png.
echo -e "\n\n## img2png: Image to png." | tee -a ~/.bashrc
echo "function img2png() {" | tee -a ~/.bashrc
echo "    path_img_src=\$1" | tee -a ~/.bashrc
echo "    path_base=\`dirname \${path_img_src}\`" | tee -a ~/.bashrc
echo "    fiename_src=\`basename \${path_img_src}\`" | tee -a ~/.bashrc
echo "    name_src=\"\${fiename_src%.*}\"" | tee -a ~/.bashrc
echo "    extension_src=\".\${fiename_src##*.}\"" | tee -a ~/.bashrc
echo "    path_img_dst=\"\${path_base}/\${name_src}.png\"" | tee -a ~/.bashrc
echo "" | tee -a ~/.bashrc
echo "    if [ \"\${extension_src}\" != \".png\" ]; then" | tee -a ~/.bashrc
echo "        convert \"\${path_img_src}\" \"\${path_img_dst}\"" | tee -a ~/.bashrc
echo "    fi" | tee -a ~/.bashrc
echo "}" | tee -a ~/.bashrc
#
#
# Remove the given file and open it via vim.
echo -e "\n\n## Remove the given file and open it via vim." | tee -a ~/.bashrc
echo "function rmvi() {" | tee -a ~/.bashrc
echo "    rm -f \$1 && vi \$1" | tee -a ~/.bashrc
echo "}" | tee -a ~/.bashrc
#
#
# Set alias, open.
echo -e "\n\n## Set alias, open." | tee -a ~/.bashrc
echo "alias open='xdg-open '" | tee -a ~/.bashrc