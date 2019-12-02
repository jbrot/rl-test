{-# LANGUAGE DataKinds, ScopedTypeVariables, TypeFamilies, TypeOperators #-}
module NN where

import Control.Monad.IO.Class
import Data.Maybe (fromJust)
import Data.Singletons
import Data.Singletons.Prelude.Bool
import Data.Singletons.Prelude.Num
import Data.Singletons.TypeLits
import Data.Type.Equality ((:~:)(..))
import qualified Data.Vector.Storable as V
import Grenade
import Numeric.LinearAlgebra.Static
import System.Random

type NL = '[ FullyConnected 4 128, Relu, FullyConnected 128 2, Softmax ]
type NNet = Network NL '[ 'D1 4, 'D1 128, 'D1 128, 'D1 2, 'D1 2 ]
type Grad = Gradients NL

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

seedVector :: (KnownNat n, (1 <=? n) ~ 'True) => MonadIO m => R n -> m (Int, R n)
seedVector v = fmap (\t -> (t, fromJust . create . flip V.unsafeUpd [(t, 1 / ((unwrap v) V.! t))] $ V.replicate (size v) 0)) (sample v)

apply :: MonadIO m => NNet -> R 4 -> m (Int, Gradients NL)
apply nn v = fmap (fmap (fst . runGradient nn tape . S1D)) (seedVector o)
    where (tape, S1D o) = runNetwork nn (S1D v)
