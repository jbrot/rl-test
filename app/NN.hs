{-# LANGUAGE DataKinds, GADTs, ScopedTypeVariables, TypeOperators #-}
module NN where

import Control.Monad.IO.Class
import Data.Proxy
import Data.Singletons
import Data.Singletons.Prelude.Bool
import Data.Singletons.Prelude.Num
import Data.Singletons.TypeLits
import Data.Type.Equality ((:~:)(..))
import Grenade
import Numeric.LinearAlgebra.Static
import System.Random

type NNet = Network '[ FullyConnected 4 128, Relu, FullyConnected 128 2, Softmax ] '[ 'D1 4, 'D1 128, 'D1 128, 'D1 2, 'D1 2 ]

randNN :: IO NNet
randNN = randomNetwork

sample :: (MonadIO m, KnownNat n, (1 <=? n) ~ 'True) => R n -> m Int
sample v = fmap (go v) . liftIO . randomRIO $ (0,1)
    where go :: forall n1. (KnownNat n1, (1 <=? n1) ~ 'True) => R n1 -> Double -> Int
          go vec v = if v < h
                        then 0
                        else case (SNat :: SNat n1) %- (SNat :: SNat 1) of
                               SNat -> case (SNat :: SNat 1) %<=? singByProxy t of
                                         STrue -> 1 + go t (v - h)
                                         SFalse -> 0
              where (h,t) = headTail vec

apply :: NNet -> R 4 -> R 2
apply nn = (\(S1D v) -> v) . snd . runNetwork nn . S1D
