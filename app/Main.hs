{-# LANGUAGE DataKinds, ScopedTypeVariables, TypeFamilies, TypeOperators #-}
module Main where

import Control.Monad.IO.Class
import Control.Monad.Trans.Control
import Control.Monad.Trans.State
import Data.VectorSpace
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

updateNet :: IState Cartpole -> IState Cartpole
updateNet st = st{rollout = [], nn = applyUpdate (LearningParameters 0.05 0.5 0.005) (nn st) upd}
  where gamma = 0.9
        (_,gtrl) = foldl (\(v,ls) (r,g) -> let nv = gamma * v + r in (nv, (nv,g):ls)) (0, []) (rollout st)
        average :: Fractional n => [n] -> n
        average = (/) <$> sum <*> (realToFrac . length)
        avg :: Float
        avg = average (fmap fst gtrl) 
        stdev :: Float
        stdev = sqrt . average . fmap (\(x,_) -> (x - avg)^2) $ gtrl
        upd = foldr (\(x,g) ag -> ag ^+^ ((rtf $ (avg - x) / (stdev + 1e-9)) *^ g)) zeroV gtrl

stp :: ViewPort -> Float -> IState Cartpole -> IO (IState Cartpole)
stp _ _ = execStateT $ do
    CObs x1 x2 x3 x4 <- obs <$> get
    (act, grad) <- flip apply (vec4 (rtf x1) (rtf x2) (rtf x3) (rtf x4)) . nn =<< get
    (o, r, d) <- istate (step $ case act of
                                  0 -> CLeft
                                  1 -> CRight
                                  _ -> undefined)
    o <- if d then get >>= \s -> liftIO (putStrLn $ "Reset: " ++ (show . length . rollout $ s)) >> modify updateNet >> istate reset else pure o
    modify (\s -> s{obs = o, rollout = (r,grad):(rollout s)})

main :: IO ()
main = do
    st <- defSt
    simulateIO (InWindow "Cartpole" (600,400) (10,10)) (makeColor 1 1 1 1) 500 st rnd stp
