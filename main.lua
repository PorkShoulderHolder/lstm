--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("------------------------")
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end

require('io')
require('torch')
require('nngraph')
require('base')
require('xlua')
dofile('a4_communication_loop.lua')




local cmd = torch.CmdLine()
cmd:option('-length','','how long to run')
cmd:option('-device', 1, 'which gpu')
cmd:option('-mode', "generate", 'which mode')
cmd:option('-model','model82.net','which model to load')
cmd:option('-saveas','model.net','where to save the model')
cmd:option('-random', 'no', 'no prompt')
cmd:option('-atom','word','is the model word based or character based')
cmd:option('-charmode',"default", 'set to babble if you want made up words')

cmd:option('-batch_size', 20,'')
cmd:option('-seq_length', 50,'')
cmd:option('-layers', 2, '')
cmd:option('-decay', 2, '')
cmd:option('-rnn_size', 200, '')
cmd:option('-dropout', 0.0, '')
cmd:option('-init_weight', 0.1, '')
cmd:option('-lr', 1, '')
cmd:option('-max_epoch', 14,'')
cmd:option('-max_max_epoch',55,'')
cmd:option('-max_grad_norm', 10,'')
cmd:option('-prompt','', 'prompt')

options = cmd:parse(arg or {})

ptb = require('data')

if options["atom"] == "char" then
   trainfn = "./data/ptb.char.train.txt"
   validfn = "./data/ptb.char.valid.txt"
end

local params = nil
local sl = 20
  local out_dimension = 10000
  if options["atom"] == "char" then
    sl = 50
    out_dimension = 50
  end

  params = {batch_size=options["batch_size"],
                  seq_length=options["seq_length"],
                  layers=options["layers"],
                  decay=options["decay"],
                  rnn_size=options["rnn_size"],
                  dropout=options["dropout"],
                  init_weight=options["init_weight"],
                  lr=options["lr"],
                  vocab_size=out_dimension,
                  max_epoch=options["max_epoch"],
                  max_max_epoch=options["max_max_epoch"],
                  max_grad_norm=options["max_grad_norm"]}


-- kept some original settings for posterity and reference

if options["length"] == "day" then
  -- Train 1 day and gives 82 perplexity.
  params = {batch_size=20,
                  seq_length=options["seq_length"],
                  layers=2,
                  decay=1.15,
                  rnn_size=1500,
                  dropout=0.65,
                  init_weight=0.04,
                  lr=1,
                  vocab_size=out_dimension,
                  max_epoch=14,
                  max_max_epoch=55,
                  max_grad_norm=10}
elseif options["length"] == "hour" then

  -- Trains 1h and gives test 115 perplexity.
  
  params = {batch_size=20,
                  seq_length=sl,
                  layers=2,
                  decay=1.09,
                  rnn_size=200,
                  dropout=0.1,
                  init_weight=0.1,
                  lr=options["lr"],
                  vocab_size=out_dimension,
                  max_epoch=2,
                  max_max_epoch=53,
                  max_grad_norm=5}
end

local stopwords = {}

if options["charmode"] ~= "babble" then
  stopwords[27] = true;
end

function init_states()
  
  -- setup states / load appropriate data
  
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
  local states = {}
  if options["atom"] ~= "char" then
    state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
    states = {state_train, state_valid, state_test}
    
  else
    states = {state_train, state_valid}
  end

  for _, state in pairs(states) do
    reset_state(state)
  end
end

function transfer_data(x)
  return x:cuda()
end

model = {}
last_perplexity = 100000
derr = transfer_data(torch.ones(1))
dpred = transfer_data(torch.zeros(params.vocab_size, params.batch_size))


function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s), pred})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function setup(file)
  print("Creating a RNN LSTM network.")
  local core_network = 0
  
  if file then
    core_network = torch.load(file)
  else
    core_network = create_network()
  end
  model = torch.load(options["model"])
  core_network = model.core_network

  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  print(core_network)
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network

  

  model.rnns = g_cloneManyTimes(model.core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end


function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]

    model.err[i], model.s[i], prediction = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end


function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds, dpred})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

function predict_next( w )

   -- takes a table of strings (either chars or words) and outputs a response of specified length

   local words = w
   g_disable_dropout(model.rnns)
   g_replace_table(model.s[0], model.start_s)
   local word_indices = torch.ones(params.batch_size):cuda()
   for i = 2, (#words + words[1]) - 1 do
      
      -- substitute <unk> for unknown words

      local idx = ptb.vocab_map[words[i]]
      if i <= #words - 1 and options["charmode"] ~= "babble" then
        idx = idx or 27
      end
      word_indices[1] = idx
      _, model.s[1], prediction = unpack(model.rnns[1]:forward({word_indices,word_indices,model.s[0]}))
      g_replace_table(model.s[0], model.s[1])
      if i > #words - 1 then
         local pred_ind = torch.multinomial(torch.exp(prediction[1]):double(), 1)[1]
         
         while stopwords[pred_ind] do
            
            -- keep sampling until you reach a known word

            pred_ind = torch.multinomial(torch.exp(prediction[1]):double(), 1)[1]
         end
         local predicted_word = word_for_index(pred_ind)
         table.insert(w, predicted_word)
      end
   end
   return w
end

function predict_next_char( c )

  -- outputs character distribution specified in assignment for grading purposes

  local x = ptb.vocab_map[c]
  local letter_vect = torch.ones(params.batch_size):cuda()
  letter_vect[1] = x
  _, model.s[1], prediction = unpack(model.rnns[1]:forward({letter_vect,letter_vect,model.s[0]}))
  g_replace_table(model.s[0], model.s[1])
  return prediction[1]:double()
end

function query_sentences()
    
    -- preprocessing for generating sentences. This method is queued by the -mode generate option

    if options["random"] == "no" then
      words = read_input()
    end
    local predictions = predict_next(words)
    for i = 2, #predictions do io.write(predictions[i] .. ' ') end
    io.write('\n')
end

function query_chars()

  -- Begin a new session of character queries. Clear the state 
 
  g_disable_dropout(model.rnns)
  g_replace_table(model.s[0], model.start_s)

  if options["charmode"] == "babble" then
      local chars = {}
      if options["prompt"] == '' then
       chars = read_line_chars()
      else 
        chars = {1000}
        options["prompt"]:gsub(".",function(c) if c == " " then c = "_" end table.insert(chars,c) end)
      end
      local predictions = predict_next(chars)
      for i = 2, #predictions do
        local c = predictions[i]
        if c == "_" then c = " " end
        io.write(c) 
      end
      io.write('\n')
  else
      read_letters(predict_next_char)
  end
end

function run_valid(m)
  reset_state(state_valid)
  g_disable_dropout(m.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)

  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  local atoms_per_word = 1
  if options["atom"] == "char" then
    atoms_per_word = 5.6
  end
  print("Validation set perplexity : " .. g_f3(torch.exp((atoms_per_word * perp) / len)))

  if last_perplexity > torch.exp(perp / len) then
     last_perplexity = torch.exp(perp / len)
     torch.save(options["saveas"], m)
     print("model saved as" .. options["saveas"])
  end
  g_enable_dropout(m.rnns)
end

function run_test(m)
  reset_state(state_test)
  g_disable_dropout(m.rnns)
  local perp = 0
  local len = state_test.data:size(1)

  g_replace_table(m.s[0], m.start_s)
  for i = 1, (len - 1) do
    xlua.progress(i,len - 1)

    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    local s = m.s[i - 1]
    local atoms_per_word = 1
    if options["atom"] == "char" then
      atoms_per_word = 5.6
    end
    perp_tmp, m.s[1], prediction = unpack(m.rnns[1]:forward({x, y, m.s[0]}))
    perp = perp + perp_tmp[1]
    local index = torch.multinomial(torch.exp(prediction[1]):double(), 1)[1]

    print(word_for_index(x[1]) .. ", " .. word_for_index(y[1]) .. ", " .. word_for_index(index))
    g_replace_table(m.s[0], m.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp((atoms_per_word * perp) / (len - 1))))
  g_enable_dropout(m.rnns)
end


function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function print_status( perps, start_time, beggining_time, epoch, since_beginning )
  local atoms_per_word = 1
  if options["atom"] == "char" then
    atoms_per_word = 5.6
  end
  wps = torch.floor(total_cases / torch.toc(start_time))
     since_beginning = g_d(torch.toc(beginning_time) / 60)
     print('epoch = ' .. g_f3(epoch) ..
           ', train perp. = ' .. g_f3(torch.exp(atoms_per_word * perps:mean())) ..
           ', wps = ' .. wps ..
           ', dw:norm() = ' .. g_f3(model.norm_dw) ..
           ', lr = ' ..  g_f3(params.lr) ..
           ', since beginning = ' .. since_beginning .. ' mins.')
end

function main()
  print(options)
  g_init_gpu(options["device"])

  init_states()
  print("Network parameters:")
  print(params)
  setup()
  step = 0
  epoch = 0
  total_cases = 0
  beginning_time = torch.tic()
  start_time = torch.tic()
  print("Starting training.")
  words_per_step = params.seq_length * params.batch_size
  epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  --perps
  while epoch < params.max_max_epoch do
   perp = fp(state_train)
   if perps == nil then
     perps = torch.zeros(epoch_size):add(perp)
   end
   perps[step % epoch_size + 1] = perp
   step = step + 1
   bp(state_train)
   total_cases = total_cases + params.seq_length * params.batch_size
   epoch = step / epoch_size
   if step % torch.round(epoch_size / 10) == 10 then
     print_status(perps, start_time, beginning_time, epoch, since_beginning)
   end
   if step % epoch_size == 0 then
     run_valid(model)
     if epoch > params.max_epoch then
         params.lr = params.lr / params.decay
     end
   end
   if step % 33 == 0 then
     cutorch.synchronize()
     collectgarbage()
   end
  end
  if options["atom"] ~= "char" then
    run_test()
  end
  print("Training is over.")
end


-- switch for the various modes

if options["mode"] == "train" then
  main()
end
if options["mode"] == "load" then
  print("testing")
  print(arg)
  g_init_gpu(options["device"])
  
  init_states()
  model = torch.load(options["model"])
  if options["atom"] == "char" then
    run_valid(model)
  else
    run_test(model)
  end
end
if options["mode"] == "generate" then
 
  g_init_gpu(options["device"])
  init_states()
  model = torch.load(options["model"])
  if options["atom"] == "char" then
    query_chars()
  else
    query_sentences()
  end
end

