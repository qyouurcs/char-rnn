--[[

  Functions for loading data from disk.

--]]
--
--
local loader = {}

require 'lfs'
local debugger = require('fb.debugger')

function string.ends(String,End)
    return End=='' or string.sub(String,-string.len(End))==End
end

function string.starts(String,Start)
    return string.sub(String,1,string.len(Start))==Start
end

function loader.load_dir(data_dir, fea_len)
    local cnt = 0
    for fn in lfs.dir(data_dir) do
        if fn ~= '.' and fn ~= '..' then
            --, now we load the files.
            if not string.ends(fn, 'h5') then
                -- open the file and cnt.
                fn = paths.concat(data_dir, fn)
                local fid = io.open(fn, 'r')
                while true do
                    local line = fid:read()
                    if line == nil then break end
                    cnt = cnt + 1
                end
            end
        end
    end
    printf('cnt = %d\n', cnt)

    local fea = torch.zeros(cnt, fea_len)
    local fns = {}
    local idx = 1
    
    local cnt_fn = 0
    for fn in lfs.dir(data_dir) do
        if fn ~= '.' and fn ~= '..' then
            --, now we load the files.
            if not string.ends(fn, 'h5') then
                -- open the file and cnt.
                fn = paths.concat(data_dir, fn)
                local fid = io.open(fn, 'r')
                local fid_fea = hdf5.open(fn ..  '.h5','r')
                local fea_b = fid_fea:read('fea'):all()
                fid_fea:close()

                local i = 0
                while true do
                    local line = fid:read()
                    if line == nil then break end
                    fns[idx + i] = line
                    i = i + 1
                end
                io.close(fid)
                fea[{{idx, idx + i -1}, {}}] = fea_b:squeeze()
                cnt_fn = cnt_fn + 1
                if cnt_fn % 100 == 0 then
                    printf('cnt_fn = %d, %s\n', cnt_fn, fn)
                end
                idx = idx + i
            end
        end
    end
    return fea, fns 
end

function loader.load_caps(json_fn, save_fn)
    if path.isfile(save_fn) then
        dt = torch.load(save_fn)
        return dt
    end
    local json = require 'cjson'
    local f = io.open(json_fn,'r')
    local str = f:read('*all')
    local data = json.decode(str)
    -- Now, we just transform this one.
    local dt = {}
    -- Now, we create the vocabulary mapping.
    local unordered = {}


    for k,v in pairs(data['images']) do
        local fn = data['images'][k]['filename']
        dt[fn] = {}
        local cocoid = data['images'][k]['cocoid']
        dt[fn]['cocoid'] = cocoid
        dt[fn]['caps'] = {}
        dt[fn]['caps']['raw'] = {}
        dt[fn]['caps']['tokens'] = {}
        for s,v in pairs(data['images'][k]['sentences']) do
            dt[fn]['caps']['raw'][s] = v['raw']
            dt[fn]['caps']['tokens'][s] = v['tokens']
        end
    end
    print('Saving to ' .. save_fn .. '\n')
    torch.save(save_fn,dt)
    return dt
end
--function loader.read_embedding(vocab_path, emb_path)
--  local vocab = loader.Vocab(vocab_path)
--  local embedding = torch.load(emb_path)
--  return vocab, embedding
--end
--
--function loader.read_sentences(path, vocab)
--  local sentences = {}
--  local sentences_str = {}
--  local file = io.open(path, 'r')
--  while true do
--    local line = file:read()
--    if line == nil then break end
--    local tokens = stringx.split(line)
--    local len = #tokens
--    local sent = torch.IntTensor(len)
--    local sent_str = {}
--    for i = 1, len do
--      local token = tokens[i]
--      sent[i] = vocab:index(token)
--    end
--    sentences[#sentences + 1] = sent
--    sentences_str[#sentences_str +1] = line
--  end
--
--  file:close()
--  return sentences, sentences_str
--end
--
--function loader.read_trees(parent_path, label_path)
--  print (parent_path)
--  local parent_file = io.open(parent_path, 'r')
--  local label_file
--  if label_path ~= nil then label_file = io.open(label_path, 'r') end
--  local count = 0
--  local trees = {}
--
--  while true do
--    local parents = parent_file:read()
--    if parents == nil then break end
--    parents = stringx.split(parents)
--    for i, p in ipairs(parents) do
--      parents[i] = tonumber(p)
--    end
--
--    local labels
--    if label_file ~= nil then
--      labels = stringx.split(label_file:read())
--      for i, l in ipairs(labels) do
--        -- ignore unlabeled nodes
--        if l == '#' then
--          labels[i] = nil
--        else
--          labels[i] = tonumber(l)
--        end
--      end
--    end
--
--    count = count + 1
--    trees[count] = loader.read_tree(parents, labels)
--  end
--  parent_file:close()
--  return trees
--end
--
--function loader.read_trees_no_span(parent_path, label_path)
--  print (parent_path)
--  local parent_file = io.open(parent_path, 'r')
--  local label_file
--  if label_path ~= nil then label_file = io.open(label_path, 'r') end
--  local count = 0
--  local trees = {}
--
--  while true do
--    local parents = parent_file:read()
--    if parents == nil then break end
--    parents = stringx.split(parents)
--    for i, p in ipairs(parents) do
--      parents[i] = tonumber(p)
--    end
--
--    local labels
--    if label_file ~= nil then
--      labels = stringx.split(label_file:read())
--      for i, l in ipairs(labels) do
--        -- ignore unlabeled nodes
--        if l == '#' then
--          labels[i] = nil
--        else
--          labels[i] = tonumber(l)
--        end
--      end
--    end
--
--    count = count + 1
--    trees[count] = parents
--  end
--  parent_file:close()
--  return trees
--end
--
--
--function loader.read_tree(parents, labels)
--  local size = #parents
--  local trees = {}
--  if labels == nil then labels = {} end
--  local root
--  for i = 1, size do
--    if not trees[i] and parents[i] ~= -1 then
--      local idx = i
--      local prev = nil
--      while true do
--        local parent = parents[idx]
--        if parent == -1 then
--          break
--        end
--
--        local tree = loader.Tree()
--        if prev ~= nil then
--          tree:add_child(prev)
--        end
--        trees[idx] = tree
--        tree.idx = idx
--        tree.gold_label = labels[idx]
--        if trees[parent] ~= nil then
--          trees[parent]:add_child(tree)
--          break
--        elseif parent == 0 then
--          root = tree
--          break
--        else
--          prev = tree
--          idx = parent
--        end
--      end
--    end
--  end
--
--  -- index leaves (only meaningful for constituency trees)
--  local leaf_idx = 1
--  for i = 1, size do
--    local tree = trees[i]
--    if tree ~= nil and tree.num_children == 0 then
--      tree.leaf_idx = leaf_idx
--      leaf_idx = leaf_idx + 1
--    end
--  end
--  return root
--end
--
----[[
--
--  Semantic Relatedness
--
----]]
--
--function loader.read_relatedness_dataset(dir, vocab, constituency)
--  local dataset = {}
--  dataset.vocab = vocab
--  if constituency then
--    dataset.ltrees = loader.read_trees(dir .. 'a.cparents')
--    dataset.rtrees = loader.read_trees(dir .. 'b.cparents')
--  else
--    dataset.ltrees = loader.read_trees(dir .. 'a.parents')
--    dataset.rtrees = loader.read_trees(dir .. 'b.parents')
--  end
--  dataset.lsents = loader.read_sentences(dir .. 'a.toks', vocab)
--  dataset.rsents = loader.read_sentences(dir .. 'b.toks', vocab)
--  dataset.size = #dataset.ltrees
--  local id_file = torch.DiskFile(dir .. 'id.txt')
--  local sim_file = torch.DiskFile(dir .. 'sim.txt')
--  dataset.ids = torch.IntTensor(dataset.size)
--  dataset.labels = torch.Tensor(dataset.size)
--  for i = 1, dataset.size do
--    dataset.ids[i] = id_file:readInt()
--    dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1)
--  end
--  id_file:close()
--  sim_file:close()
--  return dataset
--end
--
----[[
--
-- Sentiment
--
----]]
--
--function loader.read_sentiment_dataset(dir, vocab, fine_grained, dependency)
--  local dataset = {}
--  dataset.vocab = vocab
--  dataset.fine_grained = fine_grained
--  local trees
--  if dependency then
--    trees = loader.read_trees(dir .. 'dparents.txt', dir .. 'dlabels.txt')
--  else
--    trees = loader.read_trees(dir .. 'parents.txt', dir .. 'labels.txt')
--    for _, tree in ipairs(trees) do
--      set_spans(tree)
--    end
--  end
--
--  local sents = loader.read_sentences(dir .. 'sents.txt', vocab)
--  if not fine_grained then
--    dataset.trees = {}
--    dataset.sents = {}
--    for i = 1, #trees do
--      if trees[i].gold_label ~= 0 then
--        table.insert(dataset.trees, trees[i])
--        table.insert(dataset.sents, sents[i])
--      end
--    end
--  else
--    dataset.trees = trees
--    dataset.sents = sents
--  end
--
--  dataset.size = #dataset.trees
--  dataset.labels = torch.Tensor(dataset.size)
--  for i = 1, dataset.size do
--    remap_labels(dataset.trees[i], fine_grained)
--    dataset.labels[i] = dataset.trees[i].gold_label
--  end
--  return dataset
--end
--
----[[
--  This is for getty.
----]]
--function loader.read_getty_dataset(dir, vocab, dependency)
--  local dataset = {}
--  dataset.vocab = vocab
--
--  local trees
--  if dependency then
--    trees = loader.read_trees(dir .. 'str.dparents')
--  else
--    --trees = loader.read_trees(dir .. 'str.cparents')
--    trees = loader.read_trees_no_span(dir .. 'fns1.cparents')
--  end
--
--  local sents = loader.read_sentences(dir .. 'fns1.toks', vocab)
--  dataset.trees = trees
--  dataset.sents = sents
--
--  dataset.size = #dataset.trees
--  dataset.labels = torch.FloatTensor(dataset.size)
--
--  --local label_file = torch.DiskFile(dir .. 'label.txt')
--  local label_file = torch.DiskFile(dir .. 'fns1_label.txt')
--  for i = 1, dataset.size do
--    dataset.labels[i] = label_file:readFloat()
--  end
--
--  local map_id2fn = {}
--  local idx = 1
--  local fid = io.open( dir .. 'fns1_img_list.txt')
--  for line in fid:lines() do
--      map_id2fn[idx] = line
--      idx = idx + 1
--  end
--  
--  label_file:close()
--  -- Now, load the cnn features.
--  local cnn_fea = torch.load(dir .. 'fea1.t7')
--  cnn_fea = cnn_fea:double()
--  local cnn_fea_fns = torch.load(dir .. 'fns1.t7')
--  local map_fn2id = {}
--  for k, fn in ipairs(cnn_fea_fns) do
--      map_fn2id[path.basename(fn)] = k
--  end
--  dataset.map_fn2id = map_fn2id
--  dataset.map_id2fn = map_id2fn
--  dataset.cnn_fea = cnn_fea
--  dataset.cnn_fea_fns = cnn_fea_fns
--
--  return dataset
--end
--
--function loader.read_getty_dataset(dir, vocab, dependency, idx_fn)
--  local dataset = {}
--  dataset.vocab = vocab
--
--  local trees
--  if dependency then
--    trees = loader.read_trees_no_span(dir .. 'fns' .. idx_fn .. '.parents')
--  else
--    --trees = loader.read_trees(dir .. 'str.cparents')
--    trees = loader.read_trees_no_span(dir .. 'fns' .. idx_fn .. '.cparents')
--  end
--
--  local sents = loader.read_sentences(dir .. 'fns' .. idx_fn .. '.toks', vocab)
--  dataset.trees = trees
--  dataset.sents = sents
--
--  dataset.size = #dataset.trees
--  dataset.labels = torch.FloatTensor(dataset.size)
--
--  --local label_file = torch.DiskFile(dir .. 'label.txt')
--  local label_file = torch.DiskFile(dir .. 'fns' .. idx_fn .. '_label.txt')
--  for i = 1, dataset.size do
--    dataset.labels[i] = label_file:readFloat()
--  end
--
--  local map_id2fn = {}
--  local idx = 1
--  local fid = io.open( dir .. 'fns' .. idx_fn .. '_img_list.txt')
--  for line in fid:lines() do
--      map_id2fn[idx] = line
--      idx = idx + 1
--  end
--  
--  label_file:close()
--  -- Now, load the cnn features.
--  local cnn_fea = torch.load(dir .. 'fea' .. idx_fn .. '.t7')
--  cnn_fea = cnn_fea:double()
--  local cnn_fea_fns = torch.load(dir .. 'fns' .. idx_fn .. '.t7')
--  local map_fn2id = {}
--  for k, fn in ipairs(cnn_fea_fns) do
--      map_fn2id[path.basename(fn)] = k
--  end
--  dataset.map_fn2id = map_fn2id
--  dataset.map_id2fn = map_id2fn
--  dataset.cnn_fea = cnn_fea
--  dataset.cnn_fea_fns = cnn_fea_fns
--
--  return dataset
--end
--function loader.read_getty_dataset_t(dir, vocab, dependency)
--  -- The first step is just load the text data.
--  local dataset = {}
--  dataset.vocab = vocab
--  local trees
--  if dependency then
--    trees = loader.read_trees_no_span(dir .. 'str' .. '.parents')
--  else
--    trees = loader.read_trees_no_span(dir .. 'str' .. '.cparents')
--  end
--
--  local sents,sents_str = loader.read_sentences(dir .. 'str' .. '.toks', vocab)
--  dataset.trees = trees
--  dataset.sents = sents
--  dataset.sents_str = sents_str
--
--  local map_id2fn = {}
--  local map_fn2idx = {}
--  local idx = 1
--  local fid = io.open( dir .. 'img.txt')
--  for line in fid:lines() do
--      map_id2fn[idx] = line
--      map_fn2idx[path.basename(line)] = idx
--      idx = idx + 1
--  end
--  dataset.map_id2fn = map_id2fn
--  dataset.map_fn2idx = map_fn2idx
--  dataset.t_size = idx - 1
--  
--  return dataset
--end
--
--
--function loader.read_coco_dataset_t(dir, vocab, dependency)
--  -- The first step is just load the text data.
--  local dataset = {}
--  dataset.vocab = vocab
--  local trees
--  if dependency then
--    trees = loader.read_trees_no_span(dir .. 'str' .. '.parents')
--  else
--    trees = loader.read_trees_no_span(dir .. 'str' .. '.cparents')
--  end
--
--  local sents,sents_str = loader.read_sentences(dir .. 'str' .. '.toks', vocab)
--  dataset.trees = trees
--  dataset.sents = sents
--  dataset.sents_str = sents_str
--
--  local map_id2fn = {}
--  local map_fn2idx = {}
--  local idx = 1
--  local fid = io.open( dir .. 'img.txt')
--  for line in fid:lines() do
--      map_id2fn[idx] = line
--      if map_fn2idx[path.basename(line)] == nil then
--          map_fn2idx[path.basename(line)] = {}
--      end
--      map_fn2idx[path.basename(line)][#map_fn2idx[path.basename(line)]+1] = idx
--      idx = idx + 1
--  end
--  dataset.map_id2fn = map_id2fn
--  dataset.map_fn2idx = map_fn2idx
--  dataset.t_size = idx - 1
--  
--  return dataset
--end
--
--function loader.read_coco_dataset_v(dir, idx_fn1, idx_fn2, dataset)
--  -- Now, load the cnn features, everytime we just load two batches, use one of them
--  -- to build the negative samples.
--  local cnn_fea = torch.load(dir .. 'fea' .. idx_fn1 .. '.t7')
--  cnn_fea = cnn_fea:double()
--  local cnn_fea_fns = torch.load(dir .. 'fns' .. idx_fn1 .. '.t7')
--
--  local ref_cnn_fea = torch.load(dir .. 'fea' .. idx_fn2 .. '.t7')
--  ref_cnn_fea = ref_cnn_fea:double()
--  local ref_cnn_fea_fns = torch.load(dir .. 'fns' .. idx_fn2 .. '.t7')
--
--  local nn_fn = dir .. '/nn/nn_idx' .. idx_fn1 .. '_' .. idx_fn2 .. '.t7'
--  local nn_mat = nil
--  if path.isfile(nn_fn) then
--      nn_mat = torch.load(nn_fn)
--  end
--
--  local bmap_fn2id = {}
--  local bmap_id2fn = {}
--  for k, fn in ipairs(cnn_fea_fns) do
--      bmap_fn2id[path.basename(fn)] = k
--      bmap_id2fn[k] = path.basename(fn)
--  end
--
--  local ref_bmap_fn2id = {}
--  local ref_bmap_id2fn = {}
--  for k, fn in ipairs(ref_cnn_fea_fns) do
--      ref_bmap_fn2id[path.basename(fn)] = k
--      ref_bmap_id2fn[k] = path.basename(fn)
--  end
--
--  dataset.size = cnn_fea:size(1)
--  dataset.ref_size = ref_cnn_fea:size(1)
--
--  dataset.bmap_fn2id = bmap_fn2id
--  dataset.bmap_id2fn = bmap_id2fn
--  dataset.ref_bmap_fn2id = ref_bmap_fn2id
--  dataset.ref_bmap_id2fn = ref_bmap_id2fn
--  dataset.cnn_fea = cnn_fea
--  dataset.cnn_fea_fns = cnn_fea_fns
--
--  dataset.ref_cnn_fea = ref_cnn_fea
--  dataset.ref_cnn_fea_fns = ref_cnn_fea_fns
--
--  dataset.nn_mat = nn_mat
--  return dataset
--end
--
--function loader.read_coco_dataset_v2(dir, dataset)
--  -- Now, load the cnn features, everytime we just load two batches, use one of them
--  -- to build the negative samples.
--  local cnn_fea = torch.load(dir .. 'fea1.t7')
--  cnn_fea = cnn_fea:double()
--  local cnn_fea_fns = torch.load(dir .. 'fns1.t7')
--
--  local bmap_fn2id = {}
--  local bmap_id2fn = {}
--  for k, fn in ipairs(cnn_fea_fns) do
--      bmap_fn2id[path.basename(fn)] = k
--      bmap_id2fn[k] = path.basename(fn)
--  end
--
--  dataset.size = cnn_fea:size(1)
--
--  dataset.bmap_fn2id = bmap_fn2id
--  dataset.bmap_id2fn = bmap_id2fn
--  dataset.cnn_fea = cnn_fea
--  dataset.cnn_fea_fns = cnn_fea_fns
--
--  return dataset
--end
--
--function loader.read_coco_dataset_inception(dir)
--  -- Now, load the cnn features, everytime we just load two batches, use one of them
--  -- to build the negative samples.
--  --
--  printf("inc dir %s", dir)
--  local dataset = {}
--  local inc_fea = nil
--  local inc_map_fn2id = {}
--  local inc_map_id2fn = {}
--
--  local idx = 1
--  while true do
--      local fea_fn = dir .. '/fea' .. tostring(idx) .. '.t7'
--      if path.isfile(fea_fn) then
--          local fns_fn = dir .. '/fns' .. tostring(idx) .. '.t7'
--          local fns_tab = torch.load(fns_fn)
--          local fea = torch.load(fea_fn)
--          if inc_fea then
--              inc_fea = torch.cat(inc_fea, fea, 1)
--          else
--              inc_fea = fea
--          end
--          local k = #inc_map_id2fn
--          for i, fn in ipairs(fns_tab) do
--              inc_map_fn2id[path.basename(fn)] = i + k
--              inc_map_id2fn[i+k] = path.basename(fn)
--          end
--      else
--          break
--      end
--      idx = idx + 1
--  end
--  dataset.inc_fea = inc_fea
--  dataset.inc_map_fn2id = inc_map_fn2id
--  dataset.inc_map_id2fn = inc_map_id2fn
--  return dataset
--end
--
--function loader.read_coco_dataset_v_div(dir, idx_fn1, idx_fn2, dataset, max_val)
--  -- Now, load the cnn features, everytime we just load two batches, use one of them
--  -- to build the negative samples.
--  if max_val == nil then
--      max_val = 256
--  end
--  local cnn_fea = torch.load(dir .. 'fea' .. idx_fn1 .. '.t7')
--  cnn_fea = cnn_fea:double()
--  cnn_fea = cnn_fea / max_val
--  local cnn_fea_fns = torch.load(dir .. 'fns' .. idx_fn1 .. '.t7')
--
--  local ref_cnn_fea = torch.load(dir .. 'fea' .. idx_fn2 .. '.t7')
--  ref_cnn_fea = ref_cnn_fea:double()
--  ref_cnn_fea = ref_cnn_fea / max_val
--  local ref_cnn_fea_fns = torch.load(dir .. 'fns' .. idx_fn2 .. '.t7')
--
--  local bmap_fn2id = {}
--  local bmap_id2fn = {}
--  for k, fn in ipairs(cnn_fea_fns) do
--      bmap_fn2id[path.basename(fn)] = k
--      bmap_id2fn[k] = path.basename(fn)
--  end
--
--  local ref_bmap_fn2id = {}
--  local ref_bmap_id2fn = {}
--  for k, fn in ipairs(ref_cnn_fea_fns) do
--      ref_bmap_fn2id[path.basename(fn)] = k
--      ref_bmap_id2fn[k] = path.basename(fn)
--  end
--
--  dataset.size = cnn_fea:size(1)
--  dataset.ref_size = ref_cnn_fea:size(1)
--
--  dataset.bmap_fn2id = bmap_fn2id
--  dataset.bmap_id2fn = bmap_id2fn
--  dataset.ref_bmap_fn2id = ref_bmap_fn2id
--  dataset.ref_bmap_id2fn = ref_bmap_id2fn
--  dataset.cnn_fea = cnn_fea
--  dataset.cnn_fea_fns = cnn_fea_fns
--
--  dataset.ref_cnn_fea = ref_cnn_fea
--  dataset.ref_cnn_fea_fns = ref_cnn_fea_fns
--  return dataset
--end
--
--function loader.read_coco_dataset_v2_div(dir, dataset, max_val)
--
--  -- Now, load the cnn features, everytime we just load two batches, use one of them
--  -- to build the negative samples.
--  if max_val == nil then
--      max_val = 256
--  end
--
--  local cnn_fea = torch.load(dir .. 'fea1.t7')
--  cnn_fea = cnn_fea:double()
--  cnn_fea = cnn_fea / max_val
--  local cnn_fea_fns = torch.load(dir .. 'fns1.t7')
--
--  local bmap_fn2id = {}
--  local bmap_id2fn = {}
--  for k, fn in ipairs(cnn_fea_fns) do
--      bmap_fn2id[path.basename(fn)] = k
--      bmap_id2fn[k] = path.basename(fn)
--  end
--
--  dataset.size = cnn_fea:size(1)
--
--  dataset.bmap_fn2id = bmap_fn2id
--  dataset.bmap_id2fn = bmap_id2fn
--  dataset.cnn_fea = cnn_fea
--  dataset.cnn_fea_fns = cnn_fea_fns
--
--  return dataset
--end
--
--function set_spans(tree)
--  if tree.num_children == 0 then
--    tree.lo, tree.hi = tree.leaf_idx, tree.leaf_idx
--    return
--  end
--
--  for i = 1, tree.num_children do
--    set_spans(tree.children[i])
--  end
--
--  tree.lo, tree.hi = tree.children[1].lo, tree.children[1].hi
--  for i = 2, tree.num_children do
--    tree.lo = math.min(tree.lo, tree.children[i].lo)
--    tree.hi = math.max(tree.hi, tree.children[i].hi)
--  end
--end
--
--function remap_labels(tree, fine_grained)
--  if tree.gold_label ~= nil then
--    if fine_grained then
--      tree.gold_label = tree.gold_label + 3
--    else
--      if tree.gold_label < 0 then
--        tree.gold_label = 1
--      elseif tree.gold_label == 0 then
--        tree.gold_label = 2
--      else
--        tree.gold_label = 3
--      end
--    end
--  end
--
--  for i = 1, tree.num_children do
--    remap_labels(tree.children[i], fine_grained)
--  end
--end
--
return loader
