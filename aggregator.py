import os
import re

def create_markdown(folder_path):
    # Regular expression to match the filename pattern
    filename_pattern = re.compile(r"i([0-9.]+)_r([0-9.]+)_k([0-9.]+)_([\w\d]+)\.png")
    
    # List to store the markdown lines
    markdown_lines = []
    
    # Iterate over all files in the directory
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            match = filename_pattern.match(filename)
            if match:
                float1, float2, float3, string1 = match.groups()
                if string1=='full':
                    method='continuum activation function'
                else:
                    method='discrete activation function'
                
                caption = f"**Spatio-temporal solution using {method}**. Electrode current I={float1} mA; Electrode distance d={float2} m; Axon curvature k={float3}_{string1}"
                image_path = os.path.join(folder_path, filename)
                markdown_lines.append(f"![{caption}]({image_path})\n\n")
                #markdown_lines.append(f"{caption}\n")
                #markdown_lines.append("\n")
    
    # Write the markdown file with two images per page
    for line in markdown_lines:
        print(line)
        
    markdown_content = ""
    for i in range(0, len(markdown_lines), 2):
        page_content = "".join(markdown_lines[i:i+2])
        markdown_content += page_content #+ "\\newpage\n\n"
    
    # Save the markdown content to agg.md in the same folder
    output_path = os.path.join(folder_path, "agg.md")
    with open(output_path, "w") as md_file:
        md_file.write(markdown_content)

    print(f"Markdown file created at {output_path}")

# Example usage
folder_path = "./results"
create_markdown(folder_path)

