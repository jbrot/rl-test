{-# LANGUAGE DataKinds #-}
module NN where

import GHC.TypeNats
import Numeric.LinearAlgebra.Static

data NNet = NNet { l1 :: L 128 4
                 , l2 :: L 2 128
                 }

randNN :: IO NNet
randNN = NNet <$> rand <*> rand

relu :: Double -> Double
relu x
  | x > 1 = 1
  | x < -1 = -1
  | otherwise = x

softmax :: (KnownNat n) => R n -> R n
softmax v = dvmap (/tot) scld
    where scld = dvmap exp v
          tot = scld <.> 1

apply :: NNet -> R 4 -> R 2
apply nn = softmax . ((l2 nn) #>) . dvmap relu . ((l1 nn) #>) 
