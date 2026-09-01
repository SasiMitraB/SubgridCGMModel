-- tikz.lua: A Pandoc/Quarto Lua filter to render inline TikZ code blocks directly to SVG
local function read_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local content = f:read("*all")
    f:close()
    return content
end

function CodeBlock(block)
    if block.classes:includes('tikz') or block.attributes['format'] == 'tikz' then
        local content = block.text
        local hash = pandoc.sha1(content)
        local out_dir = "tikz_cache"
        os.execute("mkdir -p " .. out_dir)

        local tex_file = out_dir .. "/tikz_" .. hash .. ".tex"
        local pdf_file = out_dir .. "/tikz_" .. hash .. ".pdf"
        local svg_file = out_dir .. "/tikz_" .. hash .. ".svg"

        -- Compile if SVG does not exist or is empty
        local svg_content = read_file(svg_file)
        if not svg_content or #svg_content == 0 then
            local full_tex
            if content:match("\\documentclass") then
                full_tex = content
            else
                local preamble = block.attributes['preamble'] or ""
                local libraries = block.attributes['libraries'] or "positioning,arrows.meta,calc,decorations.pathreplacing"
                full_tex = "\\documentclass[tikz,border=8pt]{standalone}\n" ..
                           "\\usepackage{tikz}\n" ..
                           "\\usepackage{amsmath}\n" ..
                           "\\usetikzlibrary{" .. libraries .. "}\n" ..
                           "\\definecolor{teal}{rgb}{0,0.5,0.5}\n" ..
                           preamble .. "\n" ..
                           "\\begin{document}\n" ..
                           content .. "\n" ..
                           "\\end{document}\n"
            end

            local f = io.open(tex_file, "w")
            f:write(full_tex)
            f:close()

            local cmd = string.format(
                "pdflatex -interaction=nonstopmode -output-directory=%s %s >/dev/null 2>&1 && pdftocairo -svg %s %s",
                out_dir, tex_file, pdf_file, svg_file
            )
            os.execute(cmd)
            svg_content = read_file(svg_file)
        end

        if not svg_content or #svg_content == 0 then
            return pandoc.Para({pandoc.Strong({pandoc.Str("[Error rendering TikZ block. Please check LaTeX syntax or error logs in " .. out_dir .. "]")})})
        end

        local width = block.attributes['width'] or "95%"
        local style = block.attributes['style'] or "background-color: #fdfdfd; padding: 12px; border-radius: 8px; margin: 0 auto; display: flex; justify-content: center;"

        -- Wrap SVG inside a styled div for neat alignment and background
        local html = string.format(
            '<div class="tikz-diagram" style="%s; width: %s;">\n%s\n</div>',
            style, width, svg_content
        )

        return pandoc.RawBlock('html', html)
    end
end
