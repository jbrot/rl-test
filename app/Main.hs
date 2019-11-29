{-# LANGUAGE DataKinds, ScopedTypeVariables, TypeFamilies, TypeOperators #-}
module Main where

import Control.Monad.IO.Class
import Control.Monad.Trans.Control
import Control.Monad.Trans.State
import Data.Proxy
import Data.Type.Equality ((:~:)(..))
import GHC.TypeLits.Compare (isLE)
import GHC.TypeLits.Witnesses
import GHC.TypeNats
import Graphics.Gloss.Interface.IO.Simulate
import Graphics.Gloss.Data.ViewPort
import Numeric.LinearAlgebra.Static
import System.Random

import NN
import RL.Gym

data IState g = IState { gym :: g
                       , obs :: Observation g
                       , nn :: NNet
                       }

istate :: (Monad m) => (g -> (a,g)) -> StateT (IState g) m a
istate f = do
    (out, ngym)  <- fmap (f . gym) get
    modify $ \st -> st{gym = ngym}
    pure out

defSt :: Gym g => IO (IState g)
defSt = fmap reset start >>= \(o, g) -> IState g o <$> randNN

rnd :: Gym g => IState g -> IO Picture
rnd = pure . translate (-300) (-200) . render 600 400 . gym

rtf = realToFrac

sample :: (MonadIO m, KnownNat n, 1 <= n) => R n -> m Int
sample v = fmap (go v) . liftIO . randomRIO $ (0,1)
    where go :: forall n1. (KnownNat n1, 1 <= n1) => R n1 -> Double -> Int
          go vec v = if v < h
                        then 0
                        else case (SNat :: SNat n1) %- (SNat :: SNat 1) of
                               SNat -> case isLE (Proxy :: Proxy 1) t of
                                         Just Refl -> 1 + go t (v - h)
                                         Nothing -> 0
              where (h,t) = headTail vec

stp :: ViewPort -> Float -> IState Cartpole -> IO (IState Cartpole)
stp _ _ = execStateT $ do
    CObs x1 x2 x3 x4 <- obs <$> get
    probs <- flip apply (vec4 (rtf x1) (rtf x2) (rtf x3) (rtf x4)) . nn <$> get
    act <- flip fmap (sample probs) $ \c -> case c of
                                    0 -> CLeft
                                    1 -> CRight
    (o, r, d) <- istate (step act)
    o <- if d then liftIO (putStrLn "Reset") >> istate reset else pure o
    modify (\s -> s{obs = o})

main :: IO ()
main = do
    st <- defSt
    simulateIO (InWindow "Cartpole" (600,400) (10,10)) (makeColor 1 1 1 1) 50 st rnd stp
