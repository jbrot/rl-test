{-# LANGUAGE DataKinds, GADTs #-}
module NN where

import GHC.TypeNats
import Grenade
import Numeric.LinearAlgebra.Static

type NNet = Network '[ FullyConnected 4 128, Relu, FullyConnected 128 2, Softmax ] '[ 'D1 4, 'D1 128, 'D1 128, 'D1 2, 'D1 2 ]

randNN :: IO NNet
randNN = randomNetwork

apply :: NNet -> R 4 -> R 2
apply nn = (\(S1D v) -> v) . snd . runNetwork nn . S1D
