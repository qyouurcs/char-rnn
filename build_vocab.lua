
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local build_vocab = {}
build_vocab.__index = build_vocab

function string.ends(String,End)
    return End=='' or string.sub(String,-string.len(End))==End
end

function string.starts(String,Start)
    return string.sub(String,1,string.len(Start))==Start
end

function build_vocab.create(data_dir, dst_dir, start_char, end_char)
    if start_char == nil then
        start_char = '&'
    end

    if end_char == nil then
        end_char = '.'
    end

    local self = {}
    setmetatable(self, build_vocab)

    local vocab_file = path.join(dst_dir, 'vocab.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not path.exists(vocab_file)  then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(data_dir)
        local vocab_attr = lfs.attributes(vocab_file)
        if input_attr.modification > vocab_attr.modification then
            print('vocab.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        build_vocab.text_to_tensor(input_file, vocab_file, tensor_file)
    end

    print('loading data files...')
    local data = torch.load(tensor_file)
    self.vocab_mapping = torch.load(vocab_file)

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    debugger.enter()
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function build_vocab:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function build_vocab:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end

-- *** STATIC method ***
function build_vocab.build_vocab(in_fn, out_vocabfile)
    local ds = torch.load(in_fn)
    if path.isfile(out_vocabfile) then
        printf('loading vocab from %s', out_vocabfile)
        local i2v = {}
        local v2i = torch.load(out_vocabfile)
        for v,i in pairs(v2i) do
            i2v[i] = v
        end
        return ds, v2i, i2v
    end
    local timer = torch.Timer()
    local cnt = 0
    printf('loading json dataset %s', in_fn) 
    local ds = torch.load(in_fn)
    local tot_len = 0

    print('creating vocabulary mapping...')
    local unordered = {}
    for img, val in pairs(ds) do
        -- Now, we load each image .
        local img_meta = val
        local raw_txt = img_meta['caps']['raw']
        for cid, txt in pairs(raw_txt) do
            local str = raw_txt[cid]
            local str_lower = string.lower(str)
            for char in str_lower:gmatch'.' do
                if not unordered[char] then unordered[char] = true end
                tot_len = tot_len + 1
            end
        end
    end
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered)
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    printf('Saving the vocabulary to %s', out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    return ds, vocab_mapping, ordered
end
-- *** STATIC method ***
function build_vocab.text_to_tensor(in_textdir, out_vocabfile, out_tensorfile)
    local timer = torch.Timer()
    local cnt = 0

    print('loading text file...')
    print('creating vocabulary mapping...')
    local cache_len = 10000
    local rawdata
    local tot_len = 0
    local unordered = {}

    for fn in lfs.dir(data_dir) do
        if fn ~= '.'  and fn ~= '..' then
            if not string.ends(fn, 'h5') then
                fn = paths.concat(data_dir, fn)
                local f = assert(io.open(fn, "r"))
                rawdata = f:read(cache_len)

                repeat
                    for char in rawdata:gmatch'.' do
                        if not unordered[char] then unordered[char] = true end
                    end
                    tot_len = tot_len + #rawdata
                    rawdata = f:read(cache_len)
                until not rawdata
                f:close()

            end
        end
    end

    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
    f = assert(io.open(in_textfile, "r"))
    local currlen = 0
    rawdata = f:read(cache_len)
    repeat
        for i=1, #rawdata do
            data[currlen+i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
        end
        currlen = currlen + #rawdata
        rawdata = f:read(cache_len)
    until not rawdata
    f:close()

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
end

return build_vocab
