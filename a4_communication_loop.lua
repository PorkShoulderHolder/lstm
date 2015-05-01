stringx = require('pl.stringx')
require 'io'


function readline()

  -- Removed the unknown word handler because this is dealt with internally 
  -- Instead we replace the word with <unk> in predict_next in main.lua [line 257]
  
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  return line
end

function read_input()

  -- reads an integer denoting the length of the desired composition and a prompt of arbitrary length

  while true do
    print("Query: len word1 word2 etc")
    local ok, line = pcall(readline)
    if not ok then
      if line.code == "EOF" then
        break -- end loop
      elseif line.code == "init" then
        print("Start with a number")
      else
        print(line)
        print("Failed, try again")
      end
    else
      return line;
    end
  end
end

function read_line_chars()
  
  -- reads character prompt
  local talk_for = 300
  io.write('rdy 2 babble for ' .. talk_for .. ' chars\n')
  io.flush()
  while true do
    local t = {300}
    local line = io.read("*line")
    if line == nil then error({code="EOF"}) end
    
    line:gsub(".",function(c) if c == " " then c = "_" end table.insert(t,c) end)
    return t
  end
end

function read_letters( prediction_func )

  -- one letter at a time
  -- prediction_func takes a letter and outputs a tensor that is the nll distribution of possible followers

  io.write('OK GO\n')
  io.flush()
  while true do
    local letter = io.read("*line")
    if letter == nil then
      return 
    end
    if not #letter == 1 then
      print('one letter at a time please')
      return
    else
      local distr = prediction_func(letter)
      for i=1, 50 do
        local spacer = ' '
        if i == #distr then
          spacer = ''
        end
        io.write(distr[i] .. spacer)
      end
      io.write('\n')
      io.flush()
    end
  end
end