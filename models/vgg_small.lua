-- from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/vggbn.lua
require 'nn'

function createModel(opt)
   
   local opt = opt or {}
   local nbClasses = opt.nbClasses or 43
   local nbChannels = opt.nbChannels or 3

   -- Create tables describing VGG configurations
   cfg = {32, 32, 'M', 64, 64, 'M',128, 128, 'M'}
  
   local features = nn.Sequential()
   do
      local iChannels = nbChannels;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            local bn = nn.SpatialBatchNormalization(oChannels)
            features:add(conv3)
            features:add(bn)
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   local classifier = nn.Sequential()
   classifier:add(nn.View(128*6*6))
   classifier:add(nn.Linear(128*6*6, 512))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.BatchNormalization(512, 1e-3))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(512, nbClasses))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model
end
