{-# LANGUAGE DataKinds, ScopedTypeVariables, TypeFamilies, TypeOperators #-}
module Main where

import Control.Monad.IO.Class
import Control.Monad.Trans.Control
import Control.Monad.Trans.State
import Data.Proxy
import Grenade
import Graphics.Gloss.Interface.IO.Simulate
import Graphics.Gloss.Data.ViewPort
import Numeric.LinearAlgebra.Static
import System.Random

import NN
import RL.Gym

data IState g = IState { gym :: g
                       , obs :: Observation g
                       , nn :: NNet
                       , rollout :: [(Float,Gradients NL)]
                       }

istate :: (Monad m) => (g -> (a,g)) -> StateT (IState g) m a
istate f = do
    (out, ngym)  <- fmap (f . gym) get
    modify $ \st -> st{gym = ngym}
    pure out

defSt :: Gym g => IO (IState g)
defSt = fmap reset start >>= \(o, g) -> IState g o <$> randNN <*> pure []

rnd :: Gym g => IState g -> IO Picture
rnd = pure . translate (-300) (-200) . render 600 400 . gym

rtf = realToFrac

updateNet :: Monad m => StateT (IState Cartpole) m ()
updateNet = modify (\s -> s{rollout = []})

stp :: ViewPort -> Float -> IState Cartpole -> IO (IState Cartpole)
stp _ _ = execStateT $ do
    CObs x1 x2 x3 x4 <- obs <$> get
    (act, grad) <- flip apply (vec4 (rtf x1) (rtf x2) (rtf x3) (rtf x4)) . nn =<< get
    (o, r, d) <- istate (step $ case act of
                                  0 -> CLeft
                                  1 -> CRight
                                  _ -> undefined)
    o <- if d then liftIO (putStrLn "Reset") >> updateNet >> istate reset else pure o
    modify (\s -> s{obs = o, rollout = (r,grad):(rollout s)})

main :: IO ()
main = do
    st <- defSt
    simulateIO (InWindow "Cartpole" (600,400) (10,10)) (makeColor 1 1 1 1) 50 st rnd stp
